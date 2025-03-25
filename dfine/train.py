import os
import json
import torch
import einops
import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from .memory import ReplayBuffer
from .configs import TrainConfig
from .models import (
    Encoder,
    Decoder,
    Dfine,
)


def train(env: gym.Env, config: TrainConfig):

    # prepare logging
    log_dir = Path(config.log_dir) / datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir / "args.json", "w") as f:
        json.dump(config.dict(), f)
    
    writer = SummaryWriter(log_dir=log_dir)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # define replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )

    # define models and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = Encoder(
        raw_obs_dim=env.observation_space.shape[0],
        obs_dim=config.obs_dim,
        hidden_dim=config.hidden_dim
    ).to(device)

    decoder = Decoder(
        raw_obs_dim=env.observation_space.shape[0],
        obs_dim=config.obs_dim,
        hidden_dim=config.hidden_dim
    ).to(device)

    dfine = Dfine(
        state_dim=config.state_dim,
        action_dim=env.action_space.shape[0],
        obs_dim=config.obs_dim,
        device=device,
    ).to(device)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
        list(dfine.parameters())
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps)

    # collect experience with random actions
    for _ in range(config.num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs

    for update in range(config.num_updates):
        raw_observations, actions, _, _ = replay_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        raw_observations = torch.as_tensor(raw_observations, device=device)
        raw_observations = einops.rearrange(raw_observations, 'b l o -> l b o')
        observations = encoder(einops.rearrange(raw_observations, 'l b o -> (l b) o'))
        observations = einops.rearrange(observations, '(b l) o -> l b o', b=config.batch_size)
        actions = torch.as_tensor(actions, device=device)
        actions = einops.rearrange(actions, 'b l a -> l b a')

        mean = torch.zeros((config.batch_size, config.state_dim), device=device)
        cov = torch.eye(config.state_dim, device=device).repeat([config.batch_size, 1, 1])

        total_loss = 0

        for t in range(config.chunk_length - config.prediction_k - 1):
            mean, cov = dfine.dynamics_update(
                mean=mean,
                cov=cov,
                action=actions[t],
            )
            mean, cov = dfine.measurement_update(
                mean=mean,
                cov=cov,
                obs=observations[t+1],
            )

            pred_raw_obs = torch.zeros((config.prediction_k, config.batch_size, env.observation_space.shape[0]), device=device)

            pred_mean = mean
            pred_cov = cov

            for k in range(config.prediction_k):
                pred_mean, pred_cov = dfine.dynamics_update(
                    mean=pred_mean,
                    cov=pred_cov,
                    action=actions[t+k+1]
                )
                pred_raw_obs[k] = decoder(pred_mean @ dfine.C.T)

            true_raw_obs = raw_observations[t+2: t+2+config.prediction_k]
            true_raw_obs_flatten = einops.rearrange(true_raw_obs, "k b o -> (k b) o")
            pred_raw_obs_flatten = einops.rearrange(pred_raw_obs, "k b o -> (k b) o")

            total_loss += criterion(pred_raw_obs_flatten, true_raw_obs_flatten)

        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()

        writer.add_scalar("loss", total_loss.item(), update)
        print(f"update step: {update+1}, loss: {total_loss.item()}")

    
    torch.save(encoder.state_dict(), log_dir / "encoder.pth")
    torch.save(decoder.state_dict(), log_dir / "decoder.pth")
    torch.save(dfine.state_dict(), log_dir / "dfine.pth")

    return {"model_dir": log_dir}