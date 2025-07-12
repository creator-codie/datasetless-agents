import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, thought_vector_dim, nhead, num_encoder_layers, dim_feedforward, num_actions):
        super(PolicyNetwork, self).__init__()

        # Encoders for the two input streams
        self.state_encoder = nn.Linear(state_dim, dim_feedforward)
        self.thought_encoder = nn.Linear(thought_vector_dim, dim_feedforward)

        # Shared Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # Actor and Critic heads
        self.actor_head = nn.Linear(dim_feedforward, num_actions)
        self.critic_head = nn.Linear(dim_feedforward, 1)

    def forward(self, env_state, thought_vector):
        # Encode the inputs
        state_embedding = self.state_encoder(env_state)
        thought_embedding = self.thought_encoder(thought_vector)

        # Combine and pass through transformer
        combined = state_embedding + thought_embedding
        transformer_out = self.transformer_encoder(combined.unsqueeze(0)).squeeze(0)

        #- Actor and Critic outputs
        action_probs = torch.softmax(self.actor_head(transformer_out), dim=-1)
        state_value = self.critic_head(transformer_out)

        return action_probs, state_value

