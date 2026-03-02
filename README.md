# Optimizing AoI in Mobility-Based Vehicular Fog Networks — Dueling-DDQN

A complete Python research simulation implementing and comparing four Deep Q-Network variants for minimizing **Age of Information (AoI)** in mobility-based vehicular fog networks, based on the paper:

> **"Optimizing AoI in Mobility-Based Vehicular Fog Networks: A Dueling-DDQN Approach"**

---

## 🏗️ Project Structure

```
ML Project/
├── run.py                        # CLI entry point — trains all 4 agents
├── requirements.txt
│
├── environment/
│   ├── vehicular_env.py          # Gym-style env (reset, step, reward)
│   ├── mobility.py               # Constant-speed mobility model
│   └── aoi_model.py              # AoI formula & threshold logic
│
├── agents/
│   ├── replay_buffer.py          # Experience replay buffer
│   ├── dqn.py                    # Vanilla DQN
│   ├── ddqn.py                   # Double DQN
│   ├── dueling_dqn.py            # Dueling DQN
│   └── dueling_ddqn.py           # Dueling Double DQN (paper's method)
│
├── training/
│   └── trainer.py                # Agent-agnostic training loop
│
├── utils/
│   ├── config.py                 # All hyperparameters (dataclass)
│   └── plotting.py               # Reward, AoI, convergence plots
│
└── notebooks/
    └── experiment.ipynb          # Full experiment notebook
```

---

## 🔑 Key Concepts

### Age of Information (AoI)
Measures the freshness of information delivered from a vehicle to an RSU:

$$\Delta_{i,h} = \frac{1}{(1-\varepsilon)\lambda} + \frac{1}{(1-\varepsilon)\mu} + \frac{\lambda}{\mu(\lambda+\mu)}$$

where `λ` = offloading rate, `μ` = RSU service rate, `ε` = packet error probability.

### Mobility Model
Constant-speed model with boundary reflection:
```
x(t+1) = x(t) + v · dt · cos(θ)
y(t+1) = y(t) + v · dt · sin(θ)
```
Vehicle participates only if distance to RSU `d ≤ d_max`.

### Reward
```
r = 1 / avg_AoI
```

### Algorithms Compared

| Algorithm | Architecture | Target Computation |
|-----------|-------------|-------------------|
| DQN | Standard FC | `r + γ · max Q_target(s', a')` |
| Double DQN | Standard FC | `r + γ · Q_target(s', argmax Q_online(s', ·))` |
| Dueling DQN | V(s) + A(s,a) | `r + γ · max Q_target(s', a')` |
| **Dueling DDQN** | **V(s) + A(s,a)** | **`r + γ · Q_target(s', argmax Q_online(s', ·))`** |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all four agents (CLI)
```bash
cd "ML Project"
python run.py
```

### 3. Interactive notebook
```bash
jupyter notebook notebooks/experiment.ipynb
```

---

## ⚙️ Configuration

Edit `utils/config.py` to change any parameter:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_vehicles` | 10 | Number of vehicles N |
| `mu` | 5.0 | RSU service rate |
| `epsilon_error` | 0.1 | Packet error probability ε |
| `aoi_threshold` | 10.0 | Maximum tolerable AoI |
| `coverage_radius` | 300.0 | RSU coverage radius d_max (m) |
| `learning_rate` | 1e-3 | Adam LR |
| `gamma` | 0.99 | Discount factor |
| `num_episodes` | 1000 | Training episodes |

---

## 📊 Results

The notebook generates three plots:
1. **Reward vs Episode** — all 4 agents overlaid
2. **Average AoI vs Number of Vehicles** — scalability comparison
3. **Convergence Comparison** — smoothed AoI over training

**Dueling DDQN achieves the lowest AoI and fastest convergence** because:
- The *Dueling* decomposition `Q = V(s) + (A(s,a) - mean(A))` efficiently learns state value independent of action selection
- *Double Q-learning* prevents Q-value over-estimation, yielding more reliable λ allocations

---

## 📦 Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- Matplotlib ≥ 3.7
- Jupyter ≥ 1.0

---

## 📄 Reference

*"Optimizing AoI in Mobility-Based Vehicular Fog Networks: A Dueling-DDQN Approach"*

