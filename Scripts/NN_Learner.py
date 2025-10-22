import torch
import torch.nn as nn
import torch.nn.functional as F

class BatteryEmbedding(nn.Module):
    def __init__(self, num_batteries=4, emb_dim=6):
        super().__init__()
        self.emb = nn.Embedding(num_batteries, emb_dim)
        nn.init.uniform_(self.emb.weight, -0.05, 0.05)

    def forward(self, idx):
        return self.emb(idx)

class neural(nn.Module):
    def __init__(self, emb_dim=6, hidden_layers=3, neurons_per_layer=32):
        super().__init__()
        # Inputs: normalized t (cycle 0-1), normalized T, normalized I + emb
        in_dim = 3 + emb_dim
        layers = [nn.Linear(in_dim, neurons_per_layer), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons_per_layer, neurons_per_layer), nn.Tanh()]
        layers.append(nn.Linear(neurons_per_layer, 1))  # output C (Ah or fraction)
        self.network = nn.Sequential(*layers)

        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, T, I, emb):
        x = torch.cat([t, T, I, emb], dim=1)
        return self.network(x)

class ParameterLearner(nn.Module):
    """
    Learns parameters k, n, Ea with bounded physical ranges for stability.
    """
    def __init__(self, emb_dim=6, hidden=64):
        super().__init__()
        in_dim = 1 + emb_dim  # C + emb
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 3)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, C, emb):
        if C.dim() == 1:
            C = C.unsqueeze(1)
        x = torch.cat([C, emb], dim=1)
        raw = self.net(x)  # (B,3)

        # Map to bounded physical ranges
        # n in (0.5, 2.0) - typical reaction orders
        n = 0.5 + 1.5 * torch.sigmoid(raw[:, 1:2])

        # Ea in (1,000, 5,000) J/mol - realistic activation energies for batteries
        Ea = 20000 + 60000 * torch.sigmoid(raw[:, 2:3])
        
        # k positive, log space for wide range, clamped for stability
        raw_k = torch.clamp(raw[:, 0:1], min=-8, max=8)
        k = F.softplus(raw[:, 0:1])

        return torch.cat([k, n, Ea], dim=1)  # (B,3)
