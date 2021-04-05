from dataclasses import dataclass


# Save metrics for viewing with tensorboard.
SAVE_METRICS_TENSORBOARD = True

# Save actor & critic parameters for viewing in tensorboard.
SAVE_PARAMETERS_TENSORBOARD = False

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

WORKSPACE_PATH = "./" 
BASE_CHECKPOINT_PATH = f"{WORKSPACE_PATH}/checkpoints/{EXPERIMENT_NAME}/"

# Default Hyperparameters
SCALE_REWARD:         float = 0.01
MIN_REWARD:           float = -1000.
HIDDEN_SIZE:          float = 128
BATCH_SIZE:           int   = 512
DISCOUNT:             float = 0.99
GAE_LAMBDA:           float = 0.95
PPO_CLIP:             float = 0.2
PPO_EPOCHS:           int   = 10
MAX_GRAD_NORM:        float = 1.
ENTROPY_FACTOR:       float = 0.
ACTOR_LEARNING_RATE:  float = 1e-4
CRITIC_LEARNING_RATE: float = 1e-4
RECURRENT_SEQ_LEN:    int = 8
RECURRENT_LAYERS:     int = 1    
ROLLOUT_STEPS:        int = 2048
PARALLEL_ROLLOUTS:    int = 8
PATIENCE:             int = 200
TRAINABLE_STD_DEV:    bool = False 
INIT_LOG_STD_DEV:     float = 0.0

@dataclass
class HyperParameters():
    scale_reward:         float = SCALE_REWARD
    min_reward:           float = MIN_REWARD
    hidden_size:          float = HIDDEN_SIZE
    batch_size:           int   = BATCH_SIZE
    discount:             float = DISCOUNT
    gae_lambda:           float = GAE_LAMBDA
    ppo_clip:             float = PPO_CLIP
    ppo_epochs:           int   = PPO_EPOCHS
    max_grad_norm:        float = MAX_GRAD_NORM
    entropy_factor:       float = ENTROPY_FACTOR
    actor_learning_rate:  float = ACTOR_LEARNING_RATE
    critic_learning_rate: float = CRITIC_LEARNING_RATE
    recurrent_seq_len:    int = RECURRENT_SEQ_LEN
    recurrent_layers:     int = RECURRENT_LAYERS 
    rollout_steps:        int = ROLLOUT_STEPS
    parallel_rollouts:    int = PARALLEL_ROLLOUTS
    patience:             int = PATIENCE
    # Apply to continous action spaces only 
    trainable_std_dev:    bool = TRAINABLE_STD_DEV
    init_log_std_dev:     float = INIT_LOG_STD_DEV

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
                         #init_log_std_dev=1., trainable_std_dev=True)
    
elif ENV == "BipedalWalkerHardcore-v2" and not ENV_MASK_VELOCITY:
    # Working :-D
#    hp = HyperParameters(batch_size=256, parallel_rollouts=8, recurrent_seq_len=8, rollout_steps=2048, patience=10000, entropy_factor=1e-4, 
#                         init_log_std_dev=-1., trainable_std_dev=True, min_reward=-1.)
    hp = HyperParameters(batch_size=1024, parallel_rollouts=32, recurrent_seq_len=8, rollout_steps=2048, patience=10000, entropy_factor=1e-4, 
                         init_log_std_dev=-1., trainable_std_dev=True, min_reward=-1., hidden_size=256)
else:
    raise NotImplementedError

@dataclass
class StopConditions():
    """
    Store parameters and variables used to stop training. 
    """
    best_reward: float = -1e6
    fail_to_improve_count: int = 0
    max_iterations: int = 1000000
 
