# Rocket Guidance via Reinforcement Learning

This project implements a reinforcement learning-based guidance system
for a simulated rocket/missile using **Proximal Policy Optimization (PPO)**.

The goal is to learn physically plausible interception behavior under
realistic thrust vector control (TVC) constraints, using only
onboard-observable signals.

---

## Problem Overview

The objective is to train an agent to guide a rocket from a fixed launch
position to intercept a moving target in a 2D environment.

**Key constraints:**
- Single control input: nozzle gimbal rate (continuous, [-1, 1])
- No throttle control: constant thrust while fuel remains
- Physics-based motion: torque → angular velocity → heading → thrust direction
- Limited observations: only physically obtainable onboard signals

---

## Project Structure

```bash
.
├── environment/   # Custom Gymnasium environment (physics + simulation)
├── rl-agent/      # Training and evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── models/        # Saved models from each curriculum stage
└── README.md
```

- **environment/** → rocket simulation, physics model, observation design  
- **rl-agent/** → PPO training and evaluation pipeline  
- **models/** → saved checkpoints from curriculum stages  

---

## Key Features

### Physics-Constrained Environment

- Thrust along rocket body axis  
- Gimbal produces torque (not direct steering)  
- Crossflow drag prevents sideways exploits  
- Gravity and fuel constraints included  

### Observation Design

Includes:
- Rocket state (position, velocity, orientation)
- Relative target position and velocity
- Distance to target
- Line-of-sight (LOS) rate
- Closing velocity
- Obstacle features

---

## Training Approach

### Algorithm

- PPO (Stable-Baselines3)
- MLP policy (64×64)
- CPU training

### Curriculum Learning

| Stage | Description           | Purpose            |
|------|----------------------|--------------------|
| 0    | Static target        | Basic intercept    |
| 1    | Static → drift       | Reacquisition      |
| 2    | Predictable motion   | Lead pursuit       |
| 3    | Motion + obstacle    | Full guidance      |

Training stopped at **Stage 3**, where stable guidance behavior was achieved.

---

## Reward Design

- Primary: progress toward target  
- Secondary: alignment, smoothness  
- Penalties: obstacle proximity, control effort  
- Terminal: strong reward for intercept  

### Key Insight

If penalties outweigh progress → the agent learns inaction.  
Fix: ensure progress dominates all penalties.

---

## Evaluation

### Quantitative

- ~90%+ hit rate (simple scenarios)  
- ~70–76% hit rate (dynamic + obstacles)  

### Human-Controlled Evaluation

- Typical survival: 50–70%  
- Best runs: ~90%  
- 100% achieved after multiple attempts  

**Interpretation:**
- The agent consistently intercepts reactive targets  
- Perfect evasion is possible but difficult  
- In most runs, the agent successfully lands at least one hit  

---

## Generalization

- Strong on trained level (Stage 3)  
- Degrades on lower levels  
- Moderate performance on unseen higher levels (~60%)  

---

## Emergent Behavior

The agent learned:
- Lead pursuit  
- Curved intercept trajectories  
- Momentum-based control  

These behaviors resemble **proportional navigation**, a classical guidance strategy.

---

## Key Challenges

### Physics Exploits
Fixed incorrect thrust modeling and sideways glide behavior.

### Reward Collapse
Resolved by enforcing reward hierarchy.

### Normalization Issues
Fixed mismatched normalization statistics.

### Curriculum Instability
Adjusted hyperparameters for stage transitions.

---

## Documentation

This project involved multiple iterations across environment design,
reward engineering, and training stability.

- **Full Development Story** → [`docs/story.md`](docs/story.md)  
- **Key Insights & Lessons** → [`docs/insights.md`](docs/insights.md)  

---

## Limitations

- No baseline comparison  
- Limited evaluation metrics  
- No sensor noise  
- 2D only  
- Partial generalization across stages  

---

## Future Work

- Add proportional navigation baseline  
- Introduce sensor noise and domain randomization  
- Extend to 3D environments  
- Improve robustness evaluation  

---

## Tech Stack

- Python  
- Gymnasium  
- Stable-Baselines3  
- NumPy  
- Pygame  

---

## Summary

This project demonstrates that:

- Environment correctness is critical  
- Reward hierarchy determines whether learning happens  
- Physically grounded signals can produce emergent behavior  

The system achieves reliable interception and provides a strong
foundation for more advanced guidance systems.