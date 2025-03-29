import torch


def compute_grammians(
    A,
    B,
    C,
    T: int,
):

    state_dim = A.shape[0]

    Wc = torch.zeros_like(A)
    Wo = torch.zeros_like(A)

    At = torch.eye(
        state_dim,
        dtype=A.dtype,
        device=A.device,
    )

    for _ in range(T):
        Wc += At @ B @ B.T @ At.T
        Wo += At.T @ C.T @ C @ At

        At = At @ A

    return Wc, Wo