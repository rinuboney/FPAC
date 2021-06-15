import random
import torch


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = [None] * capacity
        self.idx = 0
        self.size = 0

    def __len__(self):
        return self.size

    def append(self, obs, action, reward, next_obs, not_done):
        self.buffer[self.idx] = (
            torch.from_numpy(obs),
            torch.FloatTensor(action),
            torch.FloatTensor([reward]),
            torch.from_numpy(next_obs),
            torch.FloatTensor([not_done])
        )
        self.size = min(self.size+1, self.capacity)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        idxs = random.sample(range(self.size), batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        return [torch.stack(t).to(self.device) for t in zip(*batch)]

