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
    """
    Q_01, rewards_01, _ = train(
        algorithm="qlearning",
        episodes=600,
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=1.0,
    )

    Q_05, rewards_05, _ = train(
        algorithm="qlearning",
        episodes=600,
        epsilon_start=0.5,
        epsilon_end=0.5,
        epsilon_decay=1.0,
    )

    Q_decay, rewards_decay, epsilon_log = train(
        algorithm="qlearning",
        episodes=600,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(smooth(rewards_01), label="Fixed ε = 0.1")
    plt.plot(smooth(rewards_05), label="Fixed ε = 0.5")
    plt.plot(smooth(rewards_decay), label="Decaying ε")
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed reward")
    plt.title("Q-learning reward curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epsilon_log, label="Decaying ε")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon schedule")
    plt.legend()

    plt.tight_layout()
    plt.savefig("epsilon_experiment.png")
    plt.show()

    print(f"Fixed 0.1  | final 50-ep avg: {np.mean(rewards_01[-50:]):.2f}")
    print(f"Fixed 0.5  | final 50-ep avg: {np.mean(rewards_05[-50:]):.2f}")
    print(f"Decaying ε | final 50-ep avg: {np.mean(rewards_decay[-50:]):.2f}")


def experiment_algorithms():
    """
    Train Q-learning and SARSA with identical hyperparameters and compare.
    """
    Q_ql, rewards_ql, _ = train(
        algorithm="qlearning",
        episodes=600,
        alpha=0.1,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
    )

    Q_sa, rewards_sa, _ = train(
        algorithm="sarsa",
        episodes=600,
        alpha=0.1,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(smooth(rewards_ql), label="Q-learning")
    plt.plot(smooth(rewards_sa), label="SARSA")
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed reward")
    plt.title("Q-learning vs SARSA")
    plt.legend()
    plt.tight_layout()
    plt.savefig("algorithm_comparison.png")
    plt.show()

    print_policy(Q_ql, label="Q-learning Policy")
    print_policy(Q_sa, label="SARSA Policy")

    states = {
        2: "above hole",
        5: "left of hole",
        7: "right of hole",
        10: "below hole",
    }

    print("Q-values near the hole:")
    for s, label in states.items():
        print(
            f"State {s:2d} ({label:>12}) | "
            f"Q-learning max: {np.max(Q_ql[s]):6.2f} | "
            f"SARSA max: {np.max(Q_sa[s]):6.2f}"
        )

    print(f"\nQ-learning | final 50-ep avg: {np.mean(rewards_ql[-50:]):.2f}")
    print(f"SARSA      | final 50-ep avg: {np.mean(rewards_sa[-50:]):.2f}")

    # SARSA usually has lower Q-values near the hole because it learns
    # from the action it actually takes next, including exploration risk.
    # Q-learning uses the max next Q-value, so it is more optimistic.


def plot_value_heatmap(Q, title="State value function V*(s)"):
    """
    Visualize V*(s) = max_a Q(s, a) as a colour heatmap.
    """
    V = np.max(Q, axis=1)
    V_grid = V.reshape(4, 4)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(V_grid, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im)

    for row in range(4):
        for col in range(4):
            s = row * 4 + col
            val = V_grid[row, col]

            if s == GridWorld.GOAL:
                text = "G"
            elif s == GridWorld.HOLE:
                text = "H"
            else:
                text = f"{val:.1f}"

            ax.text(col, row, text, ha="center", va="center",
                    fontsize=12, fontweight="bold")

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f"col {c}" for c in range(4)])
    ax.set_yticklabels([f"row {r}" for r in range(4)])
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig("value_heatmap.png")
    plt.show()


def replay_episode(Q, epsilon=0.0, max_steps=50):
    """
    Run one episode and print a step-by-step trace.
    """
    env = GridWorld()
    state = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    while not done and step < max_steps:
        action = choose_action(state, Q, epsilon)
        row, col = env.state_to_coords(state)
        action_name = GridWorld.ACTION_NAMES[action]

        next_state, reward, done = env.step(action)

        step += 1
        total_reward += reward

        print(
            f"Step {step:2d} | State: {state:2d} (row {row}, col {col}) | "
            f"Action: {action_name:<5} | Reward: {reward:.1f}"
        )

        state = next_state

    reached_goal = (state == GridWorld.GOAL)
    print(
        f"Episode finished in {step} steps. "
        f"Total reward: {total_reward:.2f}. "
        f"Reached goal: {reached_goal}"
    )


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