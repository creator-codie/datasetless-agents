import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

class PPOTrainer:
    def __init__(self, policy_network, config):
        self.policy_network = DDP(policy_network)
        self.optimizer = Adam(self.policy_network.parameters(), lr=config['lr'])
        self.scaler = GradScaler()
        self.config = config

    def setup_distributed(self, rank, world_size):
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def train(self, batch):
        # Unpack batch
        states, thought_vectors, actions, old_log_probs, rewards, dones, values = batch

        # Compute advantages and returns
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values[:-1]

        # Training loop
        for _ in range(self.config['ppo_epochs']):
            with autocast():
                # Get new action probabilities and values
                new_action_probs, new_values = self.policy_network(states, thought_vectors)
                new_values = new_values.squeeze()

                dist = torch.distributions.Categorical(new_action_probs)
                new_log_probs = dist.log_prob(actions)

                # Ratio of new to old policies
                ratio = torch.exp(new_log_probs - old_log_probs)

                # PPO clip loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config['clip_epsilon'], 1.0 + self.config['clip_epsilon']) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                critic_loss = (returns - new_values).pow(2).mean()

                # Entropy bonus
                entropy_loss = -dist.entropy().mean()

                # Total loss
                loss = actor_loss + self.config['vf_coef'] * critic_loss + self.config['entropy_coef'] * entropy_loss

            # Gradient clipping and optimization
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer) # Unscale before clipping
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config['max_grad_norm'])
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def _compute_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards) - 1)):
            next_value = values[t + 1]
            delta = rewards[t] + self.config['gamma'] * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lam = delta + self.config['gamma'] * self.config['lambda'] * (1 - dones[t]) * last_gae_lam
        return advantages

