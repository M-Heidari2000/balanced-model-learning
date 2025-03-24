from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    seed: int = 0
    log_dir: str = "log"
    state_dim: int = 30
    hidden_dim: int = 32
    min_var: float = 1e-2
    buffer_capacity: int = 100000
    num_episodes: int = 100
    lr: float = 1e-3
    eps: float = 1e-5
    clip_grad_norm: int = 1000
    
    dict = asdict