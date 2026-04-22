# Lab: Q-Learning and SARSA on GridWorld

### gridworld.py — read only

This file contains the `GridWorld` class. It is a complete, working
environment. You will import it and call its methods, but you must not
change anything inside it. If you modify it, your results will be
unreliable and tests may fail.

### agent.py — your main implementation file

This is where all the RL algorithms live. You must complete every
section marked with `# TODO`. Do not change any function signatures
(names, arguments, return values) — the autograder and run.py depend
on them being exactly as written.

Functions you will implement:
- `choose_action`        — ε-greedy action selection
- `update_Q_learning`   — one Q-learning TD update
- `update_SARSA`        — one SARSA TD update
- `train`               — unified training loop for both algorithms

### run.py — your experiment and visualization file

This is where you run experiments, produce plots, and answer the
analysis questions. Complete every `# TODO` section. Each function
corresponds to one part of the lab.

Functions you will implement:
- `experiment_epsilon_decay()`  — Part C
- `experiment_algorithms()`     — Part D
- `plot_value_heatmap(Q)`       — Part E1
- `replay_episode(Q)`           — Part E2

---

## Setup (only do this if you don't already have the requirements)

You need Python 3 with NumPy and Matplotlib. No other libraries.

```bash
pip install numpy matplotlib
```

To check your agent runs without errors (before all TODOs are done):

```bash
python gridworld.py      # should print nothing (no errors)
python agent.py          # will print nothing until TODOs are filled
```

Once agent.py is complete, run the full experiment script:

```bash
python run.py
```

---

## The environment

A 4×4 grid. The agent starts at the top-left (S) and must reach
the goal (G) at the bottom-right while avoiding the hole (H).

```
S  .  .  .    (row 0)
.  .  H  .    (row 1)
.  .  .  .    (row 2)
.  .  .  G    (row 3)
```

States are integers 0–15 where `state = row * 4 + col`.

| Cell     | State | Reward | Ends episode? |
|----------|-------|--------|---------------|
| S (start)| 0     | —      | No (reset)    |
| G (goal) | 15    | +10    | Yes           |
| H (hole) | 6     | −10    | Yes           |
| Any other| —     | −0.1   | No            |

Actions: `0=UP  1=DOWN  2=LEFT  3=RIGHT`

Moving into a wall keeps the agent in place.

---

## Lab parts

### Part A — Environment warmup (15 min)

Before writing any algorithm code, run this in a Python shell and
make sure you understand every line of the output:

```python
from gridworld import GridWorld
env = GridWorld()
state = env.reset()
print("Start state:", state)

next_state, reward, done = env.step(1)   # move DOWN
print("After DOWN:", next_state, reward, done)

env.render()
```

Answer in your notebook:
1. What state integer encodes the hole? The goal?
2. What happens if the agent moves UP from row 0?
3. How many total (state, action) pairs are in the Q-table?

---

### Part B — Core algorithms (60 min)

Complete all four functions in `agent.py`.

**B1. `choose_action(state, Q, epsilon)`**

ε-greedy: with probability `epsilon` return a random action,
otherwise return `argmax Q[state]`.

**B2. `update_Q_learning(...)`**

One Q-learning update (off-policy). The update rule:

```
td_target        = reward + gamma * max_a' Q[next_state, a']
td_error (delta) = td_target - Q[state, action]
Q[state, action] += alpha * td_error
```

**B3. `update_SARSA(..., next_action, ...)`**

One SARSA update (on-policy). Identical to Q-learning except:

```
td_target = reward + gamma * Q[next_state, next_action]
```

Note the function takes `next_action` as an explicit argument.
This is the action the policy *actually* chose at `next_state`,
not the best possible action.

**B4. `train(algorithm, ...)`**

The unified training loop. Key structural difference:

```
Q-learning loop:           SARSA loop:
  choose action              choose FIRST action (before loop)
  take action (env.step)     loop:
  update (max next Q)          take action
  advance state                choose NEXT action  ← before update
                               update (next_action Q)
                               action = next_action
                               advance state
```

The SARSA timing is subtle: `next_action` must be chosen BEFORE
calling `update_SARSA` because the update uses its Q-value. Then
`next_action` becomes `action` for the next iteration.

Also implement ε-decay at the end of each episode:

```python
epsilon = max(epsilon_end, epsilon * epsilon_decay)
```

---

### Part C — ε-decay experiment (25 min)

Complete `experiment_epsilon_decay()` in `run.py`.

Train three Q-learning agents:
1. Fixed ε = 0.1
2. Fixed ε = 0.5
3. Decaying ε: start=1.0, end=0.01, decay=0.995

Plot smoothed reward curves for all three on the same graph.
Add a second subplot showing the epsilon schedule for the decay agent.

In your notebook:
- Which schedule learns fastest early on? Why?
- Which performs best at convergence? Why?
- Why does high fixed ε hurt late in training?

---

### Part D — Q-learning vs SARSA (30 min)

Complete `experiment_algorithms()` in `run.py`.

Train both algorithms with identical hyperparameters:
`episodes=600, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995`

Print both learned policies. Examine Q-values for states adjacent
to the hole (states 2, 5, 7, 10).

In your notebook:
1. Do the two algorithms learn the same policy or different ones?
2. Which algorithm has lower Q-values near the hole, and why?
3. Which would you trust more in a real environment where mistakes
   are costly (e.g. a robot that could fall over)? Justify.

---

### Part E — Visualization (20 min)

**E1. `plot_value_heatmap(Q)`**

Compute V*(s) = max_a Q(s, a) for all states. Reshape to 4×4
and display as a colour heatmap (red=low, green=high). Annotate
each cell with its value; mark the goal and hole with "G" and "H".

**E2. `replay_episode(Q, epsilon=0.0)`**

Run one greedy episode and print a step-by-step trace:

```
Step  1 | State:  0 (row 0, col 0) | Action: DOWN  | Reward: -0.1
Step  2 | State:  4 (row 1, col 0) | Action: RIGHT | Reward: -0.1
...
Episode finished in 7 steps. Total reward: 9.4. Reached goal: True
```

---

### Part F — Short answer

Answer each question in 2–4 sentences in your notebook:

1. Look at your value heatmap. Which states have the highest value?
   Which have the lowest (excluding terminal states)?
   Does this match your intuition about the grid layout?

2. In your SARSA implementation, `next_action` is chosen before the
   update, not after. Why does the timing matter? What would go wrong
   if you chose it after?

3. Your final ε-decay schedule reaches ε = 0.01. Why is ε = 0
   (pure greedy forever) generally a bad idea, even at convergence?

4. If you doubled the grid to 8×8 with the same relative hole
   position, what would happen to training time? What is the main
   bottleneck?

---

## Submission checklist

- [ ] `agent.py` — all TODO sections filled, no function signatures changed
- [ ] `run.py`   — all TODO sections filled
- [ ] `notebook.pdf` or `notebook.ipynb` — all plots, all written answers
- [ ] Plots saved: `epsilon_experiment.png`, `algorithm_comparison.png`, `value_heatmap.png`

---

## Common bugs

**Infinite loop in training** — most likely you forgot `state = next_state`
inside the while loop. The loop condition checks `done`, but if the state
never advances the environment never reaches a terminal state.

**SARSA doesn't improve** — check that `next_action` is chosen BEFORE
`update_SARSA` is called, and that `action = next_action` happens at
the end of the loop body (not the beginning of the next iteration).

**Q-values stay at zero** — check that `update_Q_learning` and
`update_SARSA` are actually modifying `Q` in place (with `+=`),
not just computing `td_error` without saving it.

**ValueError from np.argmax on None** — you have `action = None` in
the train loop and forgot to fill in the TODO that assigns it.
