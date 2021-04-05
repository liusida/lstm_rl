# Learned from:
# Blog post by Nikolaj Goodger, Feb 13, 2020:
# https://medium.com/@ngoodger_7766/proximal-policy-optimisation-in-pytorch-with-recurrent-models-edefb8a72180
# Souce code:
# https://gitlab.com/ngoodger/ppo_lstm/-/blob/master/recurrent_ppo.ipynb

import torch
import gym
import numpy as np
import torch.nn.functional as F
import time
import wandb

from .parameters import *
from .helper import *
from .wrapper.mask_velocity_wrapper import MaskVelocityWrapper
from .trajectory import TrajectoryDataset


class LSTMRL:
    def __init__(self, seed=0) -> None:

        batch_count = hp.parallel_rollouts * hp.rollout_steps / hp.recurrent_seq_len / hp.batch_size
        print(f"batch_count: {batch_count}")
        assert batch_count >= 1., "Less than 1 batch per trajectory.  Are you sure that's what you want?"

        # Set random seed for consistant runs.
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        # Set maximum threads for torch to avoid inefficient use of multiple cpu cores.
        torch.set_num_threads(1)
        self.train_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gather_device = "cuda" if torch.cuda.is_available() and not FORCE_CPU_GATHER else "cpu"

    def calc_discounted_return(self, rewards, discount, final_value):
        """
        Calculate discounted returns based on rewards and discount factor.
        """
        seq_len = len(rewards)
        discounted_returns = torch.zeros(seq_len)
        discounted_returns[-1] = rewards[-1] + discount * final_value
        for i in range(seq_len - 2, -1, -1):
            discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1]
        return discounted_returns

    def compute_advantages(self, rewards, values, discount, gae_lambda):
        """
        Compute General Advantage.
        """
        deltas = rewards + discount * values[1:] - values[:-1]
        seq_len = len(rewards)
        advs = torch.zeros(seq_len + 1)
        multiplier = discount * gae_lambda
        for i in range(seq_len - 1, -1, -1):
            advs[i] = advs[i + 1] * multiplier + deltas[i]
        return advs[:-1]

    def gather_trajectories(self, input_data):
        """
        Gather policy trajectories from gym environment.
        """

        _MIN_REWARD_VALUES = torch.full([hp.parallel_rollouts], hp.min_reward)
        # Unpack inputs.
        env = input_data["env"]
        actor = input_data["actor"]
        critic = input_data["critic"]

        # Initialise variables.
        obsv = env.reset()
        trajectory_data = {"states": [],
                           "actions": [],
                           "action_probabilities": [],
                           "rewards": [],
                           "true_rewards": [],
                           "values": [],
                           "terminals": [],
                           "actor_hidden_states": [],
                           "actor_cell_states": [],
                           "critic_hidden_states": [],
                           "critic_cell_states": []}
        terminal = torch.ones(hp.parallel_rollouts)

        with torch.no_grad():
            # Reset actor and critic state.
            actor.get_init_state(hp.parallel_rollouts, self.gather_device)
            critic.get_init_state(hp.parallel_rollouts, self.gather_device)
            # Take 1 additional step in order to collect the state and value for the final state.
            for i in range(hp.rollout_steps):

                trajectory_data["actor_hidden_states"].append(actor.hidden_cell[0].squeeze(0).cpu())
                trajectory_data["actor_cell_states"].append(actor.hidden_cell[1].squeeze(0).cpu())
                trajectory_data["critic_hidden_states"].append(critic.hidden_cell[0].squeeze(0).cpu())
                trajectory_data["critic_cell_states"].append(critic.hidden_cell[1].squeeze(0).cpu())

                # Choose next action
                state = torch.tensor(obsv, dtype=torch.float32)
                trajectory_data["states"].append(state)
                value = critic(state.unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                trajectory_data["values"].append(value.squeeze(1).cpu())
                action_dist = actor(state.unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
                if not actor.continuous_action_space:
                    action = action.squeeze(1)
                trajectory_data["actions"].append(action.cpu())
                trajectory_data["action_probabilities"].append(action_dist.log_prob(action).cpu())

                # Step environment
                action_np = action.cpu().numpy()
                obsv, reward, done, _ = env.step(action_np)
                terminal = torch.tensor(done).float()
                transformed_reward = hp.scale_reward * torch.max(_MIN_REWARD_VALUES, torch.tensor(reward).float())

                trajectory_data["rewards"].append(transformed_reward)
                trajectory_data["true_rewards"].append(torch.tensor(reward).float())
                trajectory_data["terminals"].append(terminal)

            # Compute final value to allow for incomplete episodes.
            state = torch.tensor(obsv, dtype=torch.float32)
            value = critic(state.unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
            # Future value for terminal episodes is 0.
            trajectory_data["values"].append(value.squeeze(1).cpu() * (1 - terminal))

        # Combine step lists into tensors.
        trajectory_tensors = {key: torch.stack(value) for key, value in trajectory_data.items()}
        return trajectory_tensors

    def split_trajectories_episodes(self, trajectory_tensors):
        """
        Split trajectories by episode.
        """

        len_episodes = []
        trajectory_episodes = {key: [] for key in trajectory_tensors.keys()}
        for i in range(hp.parallel_rollouts):
            terminals_tmp = trajectory_tensors["terminals"].clone()
            terminals_tmp[0, i] = 1
            terminals_tmp[-1, i] = 1
            split_points = (terminals_tmp[:, i] == 1).nonzero() + 1

            split_lens = split_points[1:] - split_points[:-1]
            split_lens[0] += 1

            len_episode = [split_len.item() for split_len in split_lens]
            len_episodes += len_episode
            for key, value in trajectory_tensors.items():
                # Value includes additional step.
                if key == "values":
                    value_split = list(torch.split(value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
                    # Append extra 0 to values to represent no future reward.
                    for j in range(len(value_split) - 1):
                        value_split[j] = torch.cat((value_split[j], torch.zeros(1)))
                    trajectory_episodes[key] += value_split
                else:
                    trajectory_episodes[key] += torch.split(value[:, i], len_episode)
        return trajectory_episodes, len_episodes

    def pad_and_compute_returns(self, trajectory_episodes, len_episodes):
        """
        Pad the trajectories up to hp.rollout_steps so they can be combined in a
        single tensor.
        Add advantages and discounted_returns to trajectories.
        """

        episode_count = len(len_episodes)
        padded_trajectories = {key: [] for key in trajectory_episodes.keys()}
        padded_trajectories["advantages"] = []
        padded_trajectories["discounted_returns"] = []

        for i in range(episode_count):
            single_padding = torch.zeros(hp.rollout_steps - len_episodes[i])
            for key, value in trajectory_episodes.items():
                if value[i].ndim > 1:
                    padding = torch.zeros(hp.rollout_steps - len_episodes[i], value[0].shape[1], dtype=value[i].dtype)
                else:
                    padding = torch.zeros(hp.rollout_steps - len_episodes[i], dtype=value[i].dtype)
                padded_trajectories[key].append(torch.cat((value[i], padding)))
            padded_trajectories["advantages"].append(torch.cat((self.compute_advantages(rewards=trajectory_episodes["rewards"][i],
                                                                                        values=trajectory_episodes["values"][i],
                                                                                        discount=DISCOUNT,
                                                                                        gae_lambda=GAE_LAMBDA), single_padding)))
            padded_trajectories["discounted_returns"].append(torch.cat((self.calc_discounted_return(rewards=trajectory_episodes["rewards"][i],
                                                                        discount=DISCOUNT,
                                                                        final_value=trajectory_episodes["values"][i][-1]), single_padding)))
        return_val = {k: torch.stack(v) for k, v in padded_trajectories.items()}
        return_val["seq_len"] = torch.tensor(len_episodes)

        return return_val

    def train_model(self, actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions):

        # Vector environment manages multiple instances of the environment.
        # A key difference between this and the standard gym environment is it automatically resets.
        # Therefore when the done flag is active in the done vector the corresponding state is the first new state.
        env = gym.vector.make(ENV, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT)
        if ENV_MASK_VELOCITY:
            env = MaskVelocityWrapper(env)

        while iteration < stop_conditions.max_iterations:

            actor = actor.to(self.gather_device)
            critic = critic.to(self.gather_device)
            start_gather_time = time.time()

            # Gather trajectories.
            input_data = {"env": env, "actor": actor, "critic": critic, "discount": hp.discount,
                          "gae_lambda": hp.gae_lambda}
            trajectory_tensors = self.gather_trajectories(input_data)
            trajectory_episodes, len_episodes = self.split_trajectories_episodes(trajectory_tensors)
            trajectories = self.pad_and_compute_returns(trajectory_episodes, len_episodes)

            # Calculate mean reward.
            complete_episode_count = trajectories["terminals"].sum().item()
            terminal_episodes_rewards = (trajectories["terminals"].sum(axis=1) * trajectories["true_rewards"].sum(axis=1)).sum()
            mean_reward = terminal_episodes_rewards / (complete_episode_count)

            # Check stop conditions.
            if mean_reward > stop_conditions.best_reward:
                stop_conditions.best_reward = mean_reward
                stop_conditions.fail_to_improve_count = 0
            else:
                stop_conditions.fail_to_improve_count += 1
            if stop_conditions.fail_to_improve_count > hp.patience:
                print(f"Policy has not yielded higher reward for {hp.patience} iterations...  Stopping now.")
                break

            trajectory_dataset = TrajectoryDataset(trajectories, batch_size=hp.batch_size,
                                                   device=self.train_device, batch_len=hp.recurrent_seq_len)
            end_gather_time = time.time()
            start_train_time = time.time()

            actor = actor.to(self.train_device)
            critic = critic.to(self.train_device)

            # Train actor and critic.
            for epoch_idx in range(hp.ppo_epochs):
                for batch in trajectory_dataset:

                    # Get batch
                    actor.hidden_cell = (batch.actor_hidden_states[:1], batch.actor_cell_states[:1])

                    # Update actor
                    actor_optimizer.zero_grad()
                    action_dist = actor(batch.states)   # a sequence of 8 time steps is turned into a final result.
                    # Action dist runs on cpu as a workaround to CUDA illegal memory access.
                    action_probabilities = action_dist.log_prob(batch.actions[-1, :].to("cpu")).to(self.train_device)
                    # Compute probability ratio from probabilities in logspace.
                    probabilities_ratio = torch.exp(action_probabilities - batch.action_probabilities[-1, :])
                    surrogate_loss_0 = probabilities_ratio * batch.advantages[-1, :]
                    surrogate_loss_1 = torch.clamp(probabilities_ratio, 1. - hp.ppo_clip, 1. + hp.ppo_clip) * batch.advantages[-1, :]
                    surrogate_loss_2 = action_dist.entropy().to(self.train_device)
                    actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(hp.entropy_factor * surrogate_loss_2)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), hp.max_grad_norm)
                    actor_optimizer.step()

                    # Update critic
                    critic_optimizer.zero_grad()
                    critic.hidden_cell = (batch.critic_hidden_states[:1], batch.critic_cell_states[:1])
                    values = critic(batch.states)
                    critic_loss = F.mse_loss(batch.discounted_returns[-1, :], values.squeeze(1))
                    torch.nn.utils.clip_grad.clip_grad_norm_(critic.parameters(), hp.max_grad_norm)
                    critic_loss.backward()
                    critic_optimizer.step()

            end_train_time = time.time()
            print(f"Iteration: {iteration},  Mean reward: {mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, " +
                  f"complete_episode_count: {complete_episode_count}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
                  f"Train time: {end_train_time - start_train_time:.2f}s")

            record = {
                "iteration": iteration,
                "complete_episode_count": complete_episode_count,
                "total_reward": mean_reward,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "policy_entropy": torch.mean(surrogate_loss_2),
                "actor": actor,
                "value": critic,
            }
            wandb.log(record)

            if iteration % CHECKPOINT_FREQUENCY == 0:
                save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions)
            iteration += 1

        return stop_conditions.best_reward

    def train(self):
        wandb.init(project="LSTM_RL")
        actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions = start_or_resume_from_checkpoint(device=self.train_device)
        wandb.watch([actor, critic], log="all")
        score = self.train_model(actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions)
