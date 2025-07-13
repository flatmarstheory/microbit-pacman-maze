# 🟡 Pac-Man Mango Maze: Human + AI Learning Edition

A real-time interactive maze game where **you control Pac-Man using a micro:bit tilt sensor**, while a **Q-Learning AI** continuously learns optimal behavior alongside you. Built with Pygame + micro:bit + reinforcement learning.

<img width="1170" height="700" alt="image" src="https://github.com/user-attachments/assets/2ee855f8-21a8-4d50-8479-5b25cf6f238e" />

📺 [Watch the game in action](https://www.youtube.com/watch?v=pAlz-TIt3jE)

---

## 🎯 Why This Project?

This game was built to explore:
- **Human-in-the-loop learning**: A micro:bit lets you control Pac-Man using natural tilt input.
- **Live Reinforcement Learning**: A Q-learning agent updates its knowledge constantly while playing.
- **Hybrid Intelligence**: Both the human and AI can guide Pac-Man simultaneously. AI improves based on what it learns from rewards in the environment.
- **Micro:bit Integration**: Hardware + Python + Pygame = accessible, powerful education-friendly AI platform.

---

## 🧠 How It Works

### Control
- The micro:bit’s **accelerometer** detects tilt.
- Large tilt values trigger user-controlled movement.
- If tilt is weak or absent, the AI takes over.

### AI: Q-Learning
- Q-Table maps state (grid cell) × action → value
- Each step:
  - **Reward**: +1 for eating mangoes, –0.01 for moving, –0.5 for hitting a wall
  - **Update Rule**:
    ```python
    Q[s,a] += alpha * (reward + gamma * max(Q[s'],:) - Q[s,a])
    ```
- **Exploration** is controlled by ε (epsilon): probability of taking a random action.

### Game World
- Grid-based maze
- Mangoes randomly appear and must be eaten
- Walls block movement
- Visual movement is **smoothly animated**, though logic works on the grid

---

## 🕹️ Controls
- Tilt the **micro:bit** in different directions to move Pac-Man
- If you stop tilting, the **AI continues learning and takes control**
- The two inputs are always live

---

## 📦 Installation

### Requirements
- Python 3.7+
- Pygame
- NumPy
- kaspersmicrobit (for Bluetooth communication)

### Install dependencies
```bash
pip install pygame numpy kaspersmicrobit
```

---

## 🚀 Run the Game

1. Make sure your micro:bit is powered on and paired over Bluetooth
2. Clone/download the repository
3. Run:
```bash
python game.py
```

You’ll see Pac-Man appear in a maze. Tilt the micro:bit to move. Watch the score, stats, and AI improve live!

---

## ⚙️ Tuning AI Parameters
These are found in `game.py` and control how your agent behaves:

```python
alpha = 0.7       # Learning rate: how much new info overrides old
gamma = 0.9       # Discount factor: favors long-term rewards
epsilon = 0.5     # Start fully exploratory
min_epsilon = 0.1
step_penalty = -0.01
wall_penalty = -0.5
```

You can change these to see how AI learning improves or worsens.

---

## 📊 Stats Displayed
- **Score**: mangoes eaten
- **Epsilon**: current exploration chance
- **Q-values** for the current grid cell
- **Mode**: always shows "Human+AI Learning"

---

## 🔊 Micro:bit Sound
When a mango is eaten, Pac-Man triggers a short tone using micro:bit’s PWM pin (P0).
Make sure a piezo or speaker is connected to hear it.

---

## 🎥 Demo Video
📺 [Watch the game in action](https://www.youtube.com/watch?v=pAlz-TIt3jE)

- Includes real-time micro:bit control
- Visual explanation of learning and strategy
- Shows Pac-Man switching seamlessly between human and AI control

---

## 📁 File Structure
```
├── game.py           # Main game loop and logic
├── mango.png         # Mango sprite image
└── README.md         # You're reading it
```

---

## 🧪 Ideas for Expansion
- Add ghosts (obstacles)
- Switch to Deep Q-Network with PyTorch
- Add levels, lives, or power-ups
- Visualize the Q-table heatmap live
- Allow saving/loading Q-tables

---

## 📜 License
MIT License © 2024 Rai Bahadur Singh

---

## ❤️ Contributing
Pull requests, issues, and forks are welcome. This is an education-first project—tinker freely!
