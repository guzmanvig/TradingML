import torch.nn as nn
import torch
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, windows_length):
        super().__init__()
        self.fc1 = nn.Linear(in_features=windows_length*2 + 4, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=3)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

    @staticmethod
    def convert_to_tensor(states,device):
        old_exchange, old_time, state, exchange_history, hours_history = states
        states_tensor = torch.tensor([[old_exchange, state, exchange_history],[old_time, state, hours_history]])
        states_gpu = states_tensor.to(device)
        return states_gpu
