import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F

from .parameters import hp

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev=None):
        super().__init__()
        self.lstms = nn.ModuleList()
        for i in range(hp.num_lstm):
            self.lstms.append(
                nn.LSTM(state_dim, hp.hidden_size, num_layers=hp.recurrent_layers)
            )
        self.mlps = nn.ModuleList()
        for i in range(hp.num_mlp):
            self.mlps.append(
                nn.Linear(state_dim, hp.hidden_size)
            )
        self.layer_hidden = nn.Linear(hp.hidden_size, hp.hidden_size)
        self.layer_policy_logits = nn.Linear(hp.hidden_size*(hp.num_mlp+hp.num_lstm), action_dim)
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space 
        self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones((action_dim), dtype=torch.float), requires_grad=trainable_std_dev)
        self.covariance_eye = torch.eye(self.action_dim).unsqueeze(0)
        self.hidden_cells = None
        
    def get_init_state(self, batch_size, device):
        self.hidden_cells = []
        for i in range(hp.num_lstm):
            self.hidden_cells.append(
                (torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device),
                torch.zeros(hp.recurrent_layers, batch_size,hp.hidden_size).to(device))
            )
        
    def forward(self, state, terminal=None):
        batch_size = state.shape[1]
        device = state.device
        if hp.num_lstm>0:
            if self.hidden_cells is None or batch_size != self.hidden_cells[0][0].shape[1]:
                self.get_init_state(batch_size, device)
        if terminal is not None:
            for i in range(hp.num_lstm):
                h = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cells[i]]
                self.hidden_cells[i] = h
        latents = []
        for i in range(hp.num_mlp):
            latent = self.mlps[i](state)
            latents.append(latent.squeeze(0))
        for i in range(hp.num_lstm):
            _, self.hidden_cells[i] = self.lstms[i](state, self.hidden_cells[i])
            latent = F.elu(self.layer_hidden(self.hidden_cells[i][0][-1]))
            latents.append(latent)
        latent_out = torch.cat(latents, dim=1)
        policy_logits_out = self.layer_policy_logits(latent_out)
        if self.continuous_action_space:
            cov_matrix = self.covariance_eye.to(device).expand(batch_size, self.action_dim, self.action_dim) * torch.exp(self.log_std_dev.to(device))
            # We define the distribution on the CPU since otherwise operations fail with CUDA illegal memory access error.
            policy_dist = torch.distributions.multivariate_normal.MultivariateNormal(policy_logits_out.to("cpu"), cov_matrix.to("cpu"))
        else:
            policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=1).to("cpu"))
        return policy_dist
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.layer_lstm = nn.LSTM(state_dim, hp.hidden_size, num_layers=hp.recurrent_layers)
        self.layer_hidden = nn.Linear(hp.hidden_size, hp.hidden_size)
        self.layer_value = nn.Linear(hp.hidden_size, 1)
        self.hidden_cell = None
        
    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device),
                            torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device))
    
    def forward(self, state, terminal=None):
        batch_size = state.shape[1]
        device = state.device
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size, device)
        if terminal is not None:
            self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell] #partially reset the hidden cell for those terminated episodes.
        _, self.hidden_cell = self.layer_lstm(state, self.hidden_cell)
        hidden_out = F.elu(self.layer_hidden(self.hidden_cell[0][-1]))
        value_out = self.layer_value(hidden_out)
        return value_out
