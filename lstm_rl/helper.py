import os
import pathlib
import pickle
import gym
from dotmap import DotMap
import torch
from torch import optim
import wandb

from .models import Actor, Critic
from .parameters import *

def get_env_space():
    """
    Return obsvervation dimensions, action dimensions and whether or not action space is continuous.
    """
    env = gym.make(ENV)
    continuous_action_space = type(env.action_space) is gym.spaces.box.Box
    if continuous_action_space:
        action_dim =  env.action_space.shape[0]
    else:
        action_dim = env.action_space.n 
    obsv_dim= env.observation_space.shape[0] 
    return obsv_dim, action_dim, continuous_action_space

def get_last_checkpoint_iteration():
    """
    Determine latest checkpoint iteration.
    """
    if os.path.isdir(BASE_CHECKPOINT_PATH):
        max_checkpoint_iteration = max([int(dirname) for dirname in os.listdir(BASE_CHECKPOINT_PATH)])
    else:
        max_checkpoint_iteration = 0
    return max_checkpoint_iteration

def save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, iteration):
    """
    Save training checkpoint.
    """
    checkpoint = DotMap()
    checkpoint.env = ENV
    checkpoint.env_mask_velocity = ENV_MASK_VELOCITY 
    checkpoint.iteration = iteration
    checkpoint.hp = hp
    CHECKPOINT_PATH = BASE_CHECKPOINT_PATH + f"{iteration}/"
    pathlib.Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True) 
    with open(CHECKPOINT_PATH + "parameters.pt", "wb") as f:
        pickle.dump(checkpoint, f)
    with open(CHECKPOINT_PATH + "actor_class.pt", "wb") as f:
        pickle.dump(Actor, f)
    with open(CHECKPOINT_PATH + "critic_class.pt", "wb") as f:
        pickle.dump(Critic, f)
    torch.save(actor.state_dict(), CHECKPOINT_PATH + "actor.pt")
    torch.save(critic.state_dict(), CHECKPOINT_PATH + "critic.pt")
    torch.save(actor_optimizer.state_dict(), CHECKPOINT_PATH + "actor_optimizer.pt")
    torch.save(critic_optimizer.state_dict(), CHECKPOINT_PATH + "critic_optimizer.pt")
    wandb.save(CHECKPOINT_PATH + "*.pt")

def start_or_resume_from_checkpoint(device, resume=True):
    """
    Create actor, critic, actor optimizer and critic optimizer from scratch
    or load from latest checkpoint if it exists. 
    """
    max_checkpoint_iteration = 0
    if resume:
        max_checkpoint_iteration = get_last_checkpoint_iteration()
    
    obsv_dim, action_dim, continuous_action_space = get_env_space()
    actor = Actor(obsv_dim,
                  action_dim,
                  continuous_action_space=continuous_action_space,
                  trainable_std_dev=hp.trainable_std_dev,
                  init_log_std_dev=hp.init_log_std_dev)
    critic = Critic(obsv_dim)
        
    actor_optimizer = optim.AdamW(actor.parameters(), lr=hp.actor_learning_rate)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=hp.critic_learning_rate)
            
    # If max checkpoint iteration is greater than zero initialise training with the checkpoint.
    if max_checkpoint_iteration > 0:
        actor_state_dict, critic_state_dict, actor_optimizer_state_dict, critic_optimizer_state_dict = load_checkpoint(device, max_checkpoint_iteration)
        
        actor.load_state_dict(actor_state_dict, strict=True) 
        critic.load_state_dict(critic_state_dict, strict=True)
        actor_optimizer.load_state_dict(actor_optimizer_state_dict)
        critic_optimizer.load_state_dict(critic_optimizer_state_dict)

        # We have to move manually move optimizer states to `device` manually since optimizer doesn't yet have a "to" method.
        for state in actor_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                   state[k] = v.to(device)

        for state in critic_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    return actor, critic, actor_optimizer, critic_optimizer, max_checkpoint_iteration
    
def load_checkpoint(device, iteration):
    """
    Load from training checkpoint.
    """
    CHECKPOINT_PATH = BASE_CHECKPOINT_PATH + f"{iteration}/"
    with open(CHECKPOINT_PATH + "parameters.pt", "rb") as f:
        checkpoint = pickle.load(f)
        
    assert ENV == checkpoint.env, "To resume training environment must match current settings."
    assert ENV_MASK_VELOCITY == checkpoint.env_mask_velocity, "To resume training model architecture must match current settings."
    assert hp == checkpoint.hp, "To resume training hyperparameters must match current settings."

    actor_state_dict = torch.load(CHECKPOINT_PATH + "actor.pt", map_location=torch.device(device))
    critic_state_dict = torch.load(CHECKPOINT_PATH + "critic.pt", map_location=torch.device(device))
    actor_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "actor_optimizer.pt", map_location=torch.device(device))
    critic_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "critic_optimizer.pt", map_location=torch.device(device))
    
    return (actor_state_dict, critic_state_dict,
           actor_optimizer_state_dict, critic_optimizer_state_dict)
       