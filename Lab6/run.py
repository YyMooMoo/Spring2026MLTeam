# run.py
# MODIFY THIS FILE.
# Complete every section marked with TODO.
# This is your experiment and visualization script.

import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
from agent import train, choose_action


# ── Utility functions (already implemented — do not change) ────────────────

def smooth(data, window=20):
    """Moving average smoothing for reward curves."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def print_policy(Q, label="Policy"):
    """Print the greedy policy as a 4x4 grid of arrows."""
    symbols = {0: "^", 1: "v", 2: "<", 3: ">"}
    print(f"\n{label}:")
    for row in range(4):
        line = ""
        for col in range(4):
            s = row * 4 + col
            if   s == GridWorld.GOAL: line += "G  "
            elif s == GridWorld.HOLE: line += "H  "
            else:                     line += symbols[int(np.argmax(Q[s]))] + "  "
        print(line)
    print()



def experiment_epsilon_decay():
    """
    Train three Q-learning agents with different epsilon schedules
    and compare their learning curves.

    Agents to train (all other hyperparameters identical):
        1. Fixed ε = 0.1  →  epsilon_start=0.1, epsilon_end=0.1, epsilon_decay=1.0
        2. Fixed ε = 0.5  →  epsilon_start=0.5, epsilon_end=0.5, epsilon_decay=1.0
        3. Decaying ε     →  epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995

    TODO:
        - Train all three agents using train(algorithm="qlearning", episodes=600, ...)
        - Create a figure with TWO subplots side by side (figsize=(12, 4)):
            Left subplot:
                - Plot the smoothed reward curve for all three agents
                - Label each line (use the label= argument in plt.plot)
                - Add plt.legend(), plt.xlabel("Episode"),
                  plt.ylabel("Smoothed reward"), plt.title(...)
                - Add a horizontal dashed line at y=0
            Right subplot:
                - Plot the epsilon_log for the DECAYING agent only
                - Add plt.xlabel("Episode"), plt.ylabel("Epsilon"), plt.title(...)
        - Save the figure as "epsilon_experiment.png"  (plt.savefig)
        - Print the final 50-episode average reward for each agent:
              print(f"Fixed 0.1  | final 50-ep avg: {np.mean(rewards[-50:]):.2f}")
    """
    pass



def experiment_algorithms():
    """
    Train Q-learning and SARSA with identical hyperparameters and compare.

    Hyperparameters for BOTH:
        episodes=600, alpha=0.1, gamma=0.9,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995

    TODO:
        - Train both algorithms
        - Plot smoothed reward curves on the same graph
          Save as "algorithm_comparison.png"
        - Call print_policy() for both Q-tables
        - Print Q-values for the four states adjacent to the hole:
              state 2  (above the hole)
              state 5  (left  of the hole)
              state 7  (right of the hole)
              state 10 (below the hole)
          For each state print: state number, location label,
          np.max(Q_ql[s]) and np.max(Q_sa[s])
        - Print final 50-episode average reward for both algorithms

    In a comment at the bottom of this function, answer:
        Which algorithm has lower Q-values near the hole, and why?
    """
    pass



def plot_value_heatmap(Q, title="State value function V*(s)"):
    """
    Visualize V*(s) = max_a Q(s, a) as a colour heatmap.

    TODO:
        - Compute V as a (16,) array: V[s] = np.max(Q[s])
        - Reshape V to shape (4, 4)
        - Create a figure (figsize=(5, 5))
        - Use ax.imshow(V_grid, cmap="RdYlGn", interpolation="nearest")
          (red = low value, green = high value)
        - Add plt.colorbar()
        - Loop over each (row, col):
              s   = row * 4 + col
              val = V_grid[row, col]
              If s == GridWorld.GOAL:  text = "G"
              If s == GridWorld.HOLE:  text = "H"
              Otherwise:               text = f"{val:.1f}"
              ax.text(col, row, text, ha="center", va="center",
                      fontsize=12, fontweight="bold")
        - Set tick labels:
              ax.set_xticks(range(4))
              ax.set_yticks(range(4))
              ax.set_xticklabels([f"col {c}" for c in range(4)])
              ax.set_yticklabels([f"row {r}" for r in range(4)])
        - Set ax.set_title(title)
        - plt.tight_layout()
        - Save as "value_heatmap.png"
        - plt.show()
    """
    pass



def replay_episode(Q, epsilon=0.0, max_steps=50):
    """
    Run one episode and print a step-by-step trace.

    For every step print exactly this format:
        Step  1 | State:  0 (row 0, col 0) | Action: DOWN  | Reward: -0.1
        Step  2 | State:  4 (row 1, col 0) | Action: RIGHT | Reward: -0.1
        ...

    At the end print:
        Episode finished in N steps. Total reward: X.XX. Reached goal: True/False

    Args:
        Q         (np.ndarray): Q-table to act with
        epsilon   (float):      exploration rate (0.0 = fully greedy)
        max_steps (int):        safety cap to prevent infinite loops

    TODO:
        - Create a GridWorld, call env.reset()
        - Loop until done or step >= max_steps
        - Use choose_action(state, Q, epsilon) to pick each action
        - Use env.state_to_coords(state) to get (row, col)
        - Use GridWorld.ACTION_NAMES[action] to get the action string
        - Track: step count, total_reward, whether goal was reached
          (goal reached = final state == GridWorld.GOAL)
    """
    pass



if __name__ == "__main__":

    print("=" * 55)
    print("Part C — Epsilon decay experiment")
    print("=" * 55)
    experiment_epsilon_decay()

    print("=" * 55)
    print("Part D — Q-learning vs SARSA")
    print("=" * 55)
    experiment_algorithms()

    print("=" * 55)
    print("Part E — Visualization")
    print("=" * 55)

    # Train a final agent for Parts E1 and E2
    print("Training final agent for visualization...")
    Q_final, _, _ = train(
        algorithm     = "qlearning",
        episodes      = 600,
        epsilon_start = 1.0,
        epsilon_end   = 0.01,
        epsilon_decay = 0.995,
    )

    plot_value_heatmap(Q_final)

    print("\nGreedy episode replay:")
    replay_episode(Q_final, epsilon=0.0)
