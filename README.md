# DeepLane: Parallel DQN for Autonomous Lane Keeping

This project trains a Deep Q-Network (DQN) agent to keep a simulated car centered in its lane.
It uses parallel Gymnasium environments, stable-baselines3, and a custom reward that
penalizes lane deviation, heading error, and aggressive steering.

## How to run

1. Install dependencies:
```bash
pip install gymnasium==0.29.1 stable-baselines3==2.3.2 matplotlib numpy


