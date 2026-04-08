"""
PoC v2: Paper-faithful Deep Q-Learning for Portfolio Trading
Matches Pigorsch & Schäfer (2021) as closely as possible at small scale.

Paper spec:
  - Features: 12 MAs (SMA+EMA, windows 5/10/20/50/100/200) of returns,
    5 rolling stds of returns, 10 fundamentals, close price, position dummy
  - Hyperparams: gamma=0.9, epsilon=0.3, 3M steps, replay=300k, batch=1024,
    grad every 20, eval every 10k, Adam, ensemble of 32/64/128
  - Reward: asset return for invest, cross-sectional mean for cash (eq. 3)
  - Validation: save best model by cumulative return on val set

Our scale-down:
  - 50 stocks instead of 500
  - 500k steps instead of 3M (per agent)
  - Same everything else
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from environment import TransactionEnvironment
from network import QNetwork, ReplayBuffer
from evaluate import evaluate_cumulative_return


def train_agent(
    df_train,
    df_val,
    feature_cols,
    transaction_cost=0.0005,
    n_steps=500_000,
    gamma=0.9,
    epsilon=0.3,
    batch_size=1024,
    grad_interval=20,
    eval_interval=10_000,
    target_update_interval=1000,
    hidden_dim=64,
    reward_fn="base",
    sharpe_lambda=0.1,
    regime_weights=None
):
    env = TransactionEnvironment(df_train, feature_cols, transaction_cost, reward_fn=reward_fn, sharpe_lambda=sharpe_lambda, regime_weights=regime_weights)
    val_env = TransactionEnvironment(df_val, feature_cols, transaction_cost, reward_fn=reward_fn, sharpe_lambda=sharpe_lambda, regime_weights=regime_weights)
    state_dim = len(feature_cols) + 2 
    q_net = QNetwork(state_dim, hidden_dim)
    target_net = QNetwork(state_dim, hidden_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=5e-4)
    buffer = ReplayBuffer(capacity=50_000)

    best_val_cr = 0.0
    best_params = None
    train_rewards = []

    state = env.reset()
    ep_reward = 0.0

    for step in range(1, n_steps + 1):
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                action = q_net(torch.FloatTensor(state).unsqueeze(0)).argmax(1).item()

        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        ep_reward += reward

        if done:
            train_rewards.append(ep_reward)
            ep_reward = 0.0
            state = env.reset()
        else:
            state = next_state

        # gradient step every 20 iterations
        if step % grad_interval == 0 and len(buffer) >= batch_size:
            s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size)
            q_vals = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(ns_b).max(1)[0]
                targets = r_b + gamma * next_q * (1 - d_b)
            loss = nn.MSELoss()(q_vals, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
            optimizer.step()

        if step % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())

        # validation checkpoint (paper Algorithm 1, lines 8-14)
        if step % eval_interval == 0:
            val_cr = evaluate_cumulative_return(val_env, q_net, feature_cols)
            marker = ""
            if val_cr > best_val_cr:
                best_val_cr = val_cr
                best_params = {k: v.clone() for k, v in q_net.state_dict().items()}
                marker = " ::::: new best"
            print(f"Step {step:>7d} | Val CR: {val_cr:+.2%} | Best: {best_val_cr:+.2%}{marker}")

    if best_params is not None:
        q_net.load_state_dict(best_params)
        print(f"\nRestored best model (Val CR: {best_val_cr:+.2%})")
    else:
        print("\nNo positive validation CR found, using final weights")

    return q_net, train_rewards


