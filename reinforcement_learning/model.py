import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Dueling Q-network (https://arxiv.org/abs/1511.06581)"""

    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128, seed=None):
        super(DuelingQNetwork, self).__init__()

        # Make the random initialization deterministic by providing a seed.
        if (seed is not None):
            torch.manual_seed(seed)

        # value network
        self.fc1_val = nn.Linear(state_size, hidsize1)
        self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc3_val = nn.Linear(hidsize2, 1)

        # advantage network
        self.fc1_adv = nn.Linear(state_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc3_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        val = F.relu(self.fc1_val(x))
        val = F.relu(self.fc2_val(val))
        val = self.fc3_val(val)

        # advantage calculation
        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc3_adv(adv)

        return val + adv - adv.mean()
