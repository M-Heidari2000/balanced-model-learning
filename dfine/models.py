import torch
import torch.nn as nn
from typing import Optional


class Encoder(nn.Module):
    """
        z_t -> o_t
    """

    def __init__(
        self,
        obs_dim: int,
        raw_obs_dim: int,
        hidden_dim: Optional[int]=None,
        dropout_p: float=0.4,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else raw_obs_dim * 2

        self.mlp_layers = nn.Sequential(
            nn.Linear(raw_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, raw_obs):
        return self.mlp_layers(raw_obs)
    

class Decoder(nn.Module):
    """
        o_t -> z_t
    """

    def __init__(
        self,
        obs_dim: int,
        raw_obs_dim: int,
        hidden_dim: Optional[int]=None,
        dropout_p: float=0.4,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2 * obs_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, raw_obs_dim),
        )

    def forward(self, obs):
        return self.mlp_layers(obs)


class Dfine(nn.Module):
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        obs_dim: int,
        device: str,
        min_var: float=1e-4,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.device = device
        self._min_var = min_var

        # Dynamics matrices
        self.v = nn.Parameter(torch.randn(1, self.state_dim, device=self.device))
        self.r = nn.Parameter(torch.randn(1, self.state_dim, device=self.device))
        self.B = nn.Parameter(
            torch.randn(self.state_dim, self.action_dim, device=self.device),
        )
        self.C = nn.Parameter(
            torch.randn(self.obs_dim, self.state_dim, device=self.device)
        )

        # Transition noise covariance (diagonal)
        self.ns = nn.Parameter(
            torch.randn(self.state_dim, device=self.device)
        )
        # Observation noise covariance (diagonal)
        self.no = nn.Parameter(
            torch.randn(self.obs_dim, device=device)
        )

    @property
    def A(self):
        return (
            torch.eye(self.state_dim, device=self.v.device)
            + nn.functional.tanh(self.v).T @ nn.functional.tanh(self.r)
        )

    def dynamics_update(
        self,
        mean,
        cov,
        action,
    ):
        """
            Single step dynamics update

            mean: b s
            cov: b s s
            action: b a
        """

        Ns = torch.diag(nn.functional.softplus(self.ns) + self._min_var)    # shape: s s
        new_mean = mean @ self.A.T + action @ self.B.T
        new_cov = self.A @ cov @ self.A.T + Ns

        return new_mean, new_cov
    
    def measurement_update(
        self,
        mean,
        cov,
        obs,
    ):
        """
            Single step measurement update
        
            mean: b s
            cov: b s s
            obs: b o
        """

        No = torch.diag(nn.functional.softplus(self.no) + self._min_var)    # shape: o o

        K = cov @ self.C.T @ torch.linalg.pinv(self.C @ cov @ self.C.T + No)
        new_mean = mean + ((obs - mean @ self.C.T).unsqueeze(1) @ K.transpose(1, 2)).squeeze(1)
        new_cov = (torch.eye(self.state_dim, device=self.device) - K @ self.C) @ cov

        return new_mean, new_cov