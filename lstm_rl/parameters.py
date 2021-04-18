import argparse
from dataclasses import dataclass
from yamldataclassconfig.config import YamlDataClassConfig

@dataclass
class HyperParameters(YamlDataClassConfig):
    """
    Treatment: how many different features_extractors do you want to use in parallel?
    """
    num_lstm:                       int  = 1
    num_mlp:                        int  = 0 # TODO: mlp is 1:1, but lstm is n:1, how to make them compatible?
    """
    Basic Information
    """
    env:                            str  = ""
    experiment_name:                str  = "DefaultExp"
    env_mask_velocity:              bool = False
    seed:                           int  = 0
    # multiprocess?
    asynchronous_environment:       bool = False
    # Force using CPU for gathering trajectories.
    force_cpu_gather:               bool = True
    # Save training state frequency in PPO iterations.
    checkpoint_frequency:           int  = 10
    """
    Hyperparameters of the models
    """
    scale_reward:         float = 0.01
    min_reward:           float = -1000.
    hidden_size:          int   = 128
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
    parallel_rollouts:    int = 1
    patience:             int = 200
    # Apply to continous action spaces only 
    trainable_std_dev:    bool = False
    init_log_std_dev:     float = 0.0
    # Stop Condition
    max_iterations:       int = 1000000
    # Render
    render:               bool = True

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config-yaml-path", type=str, required=True, help="Specify the yaml configure file for the experiment.")
args = parser.parse_args()
hp = HyperParameters()
hp.load(args.config_yaml_path)
