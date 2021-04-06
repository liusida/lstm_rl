from dataclasses import dataclass

from torch.random import seed

# Save training state frequency in PPO iterations.
CHECKPOINT_FREQUENCY = 10

# Step env asynchronously using multiprocess or synchronously.
ASYNCHRONOUS_ENVIRONMENT = False

# Force using CPU for gathering trajectories.
FORCE_CPU_GATHER = True

# Environment parameters
ENV = "CartPole-v1"
EXPERIMENT_NAME = "CartPole-v1"
ENV_MASK_VELOCITY = True 

BASE_CHECKPOINT_PATH = f"checkpoints/{EXPERIMENT_NAME}/"

@dataclass
class HyperParameters():
    """
    Basic Information
    """
    asynchronous_environment:       bool = ASYNCHRONOUS_ENVIRONMENT
    force_cpu_gather:               bool = FORCE_CPU_GATHER
    checkpoint_frequency:           int  = CHECKPOINT_FREQUENCY
    env:                            str  = ENV
    experiment_name:                str  = EXPERIMENT_NAME
    env_mask_velocity:              bool = ENV_MASK_VELOCITY
    seed:                           int  = 0
    """
    Hyperparameters of the models
    """
    scale_reward:         float = 0.01
    min_reward:           float = -1000.
    hidden_size:          float = 128
    batch_size:           int   = 512
    discount:             float = 0.99
    gae_lambda:           float = 0.95
    ppo_clip:             float = 0.2
    ppo_epochs:           int   = 10
    max_grad_norm:        float = 1.
    entropy_factor:       float = 0.
    actor_learning_rate:  float = 1e-4
    critic_learning_rate: float = 1e-4
    recurrent_seq_len:    int = 8
    recurrent_layers:     int = 1  
    rollout_steps:        int = 2048
    parallel_rollouts:    int = 8
    patience:             int = 200
    # Apply to continous action spaces only 
    trainable_std_dev:    bool = False
    init_log_std_dev:     float = 0.0
    # Stop Condition
    max_iterations:       int = 1000000

if ENV == "CartPole-v1" and ENV_MASK_VELOCITY:
    # Working perfectly with patience.
    hp = HyperParameters(parallel_rollouts=32, rollout_steps=512, batch_size=128, recurrent_seq_len=8)

elif ENV == "Pendulum-v0" and ENV_MASK_VELOCITY:
    # Works well.     
    hp = HyperParameters(parallel_rollouts=32, rollout_steps=200, batch_size=512, recurrent_seq_len=8,
                         init_log_std_dev=1., trainable_std_dev=True, actor_learning_rate=1e-3, critic_learning_rate=1e-3)

elif ENV == "LunarLander-v2" and ENV_MASK_VELOCITY:
    # Works well.
    hp = HyperParameters(parallel_rollouts=32, rollout_steps=1024, batch_size=512, recurrent_seq_len=8, patience=1000) 

elif ENV == "LunarLanderContinuous-v2" and ENV_MASK_VELOCITY:
    # Works well.
    hp = HyperParameters(parallel_rollouts=32, rollout_steps=1024, batch_size=1024, recurrent_seq_len=8, trainable_std_dev=True,  patience=1000)
    
elif ENV == "BipedalWalker-v2" and not ENV_MASK_VELOCITY:
    # Working :-D
    hp = HyperParameters(parallel_rollouts=8, rollout_steps=2048, batch_size=256, patience=1000, entropy_factor=1e-4,
                         init_log_std_dev=-1., trainable_std_dev=True, min_reward=-1.)
    
elif ENV == "BipedalWalkerHardcore-v2" and not ENV_MASK_VELOCITY:
    # Working :-D
    hp = HyperParameters(batch_size=1024, parallel_rollouts=32, recurrent_seq_len=8, rollout_steps=2048, patience=10000, entropy_factor=1e-4, 
                         init_log_std_dev=-1., trainable_std_dev=True, min_reward=-1., hidden_size=256)
else:
    raise NotImplementedError
