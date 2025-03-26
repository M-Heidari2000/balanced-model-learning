from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    seed: int = 0
    log_dir: str = "log"
    state_dim: int = 30
    obs_dim: int = 100
    hidden_dim: int = 32
    min_var: float = 1e-2
    dropout_p: float=0.4
    buffer_capacity: int = 100000
    num_train_episodes: int = 100
    num_test_episodes: int = 10
    test_interval: int = 10
    num_updates: int = 100
    chunk_length: int = 10
    prediction_k: int = 4
    batch_size: int = 64
    lr: float = 1e-3
    eps: float = 1e-8
    clip_grad_norm: int = 1000
    
    dict = asdict