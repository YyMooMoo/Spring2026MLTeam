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
    episodes = 600

    # 1. Train Fixed epsilon = 0.1
    Q_fix_low, rewards_fix_low, eps_fix_low = train(
        algorithm="qlearning", episodes=episodes,
        epsilon_start=0.1, epsilon_end=0.1, epsilon_decay=1.0
    )

    # 2. Train Fixed epsilon = 0.5
    Q_fix_high, rewards_fix_high, eps_fix_high = train(
        algorithm="qlearning", episodes=episodes,
        epsilon_start=0.5, epsilon_end=0.5, epsilon_decay=1.0
    )

    # 3. Train Decaying epsilon
    Q_decay, rewards_decay, eps_decay = train(
        algorithm="qlearning", episodes=episodes,
        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995
    )

    # --- Visualization ---
    plt.figure(figsize=(12, 4))

    # Left subplot: Smoothed Rewards
    plt.subplot(1, 2, 1)
    plt.plot(smooth(rewards_fix_low), label="Fixed ε=0.1")
    plt.plot(smooth(rewards_fix_high), label="Fixed ε=0.5")
    plt.plot(smooth(rewards_decay), label="Decaying ε")
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed reward")
    plt.title("Reward Curves by Epsilon Schedule")
    plt.legend()

    # Right subplot: Epsilon Log (Decaying agent only)
    plt.subplot(1, 2, 2)
    plt.plot(eps_decay, color='green')
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay (1.0 → 0.01)")

    plt.tight_layout()
    plt.savefig("epsilon_experiment.png")
    plt.show()

    # --- Print Statistics ---
    print(f"Fixed 0.1  | final 50-ep avg: {np.mean(rewards_fix_low[-50:]):.2f}")
    print(f"Fixed 0.5  | final 50-ep avg: {np.mean(rewards_fix_high[-50:]):.2f}")
    print(f"Decaying   | final 50-ep avg: {np.mean(rewards_decay[-50:]):.2f}")



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
    params = {
        "episodes": 600,
        "alpha": 0.1,
        "gamma": 0.9,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995
    }

    # 1. Train both algorithms
    Q_ql, rewards_ql, _ = train(algorithm="qlearning", **params)
    Q_sa, rewards_sa, _ = train(algorithm="sarsa", **params)

    # 2. Plot smoothed reward curves
    plt.figure(figsize=(8, 5))
    plt.plot(smooth(rewards_ql), label="Q-Learning (Off-policy)")
    plt.plot(smooth(rewards_sa), label="SARSA (On-policy)")
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.title("Q-Learning vs. SARSA Performance")
    plt.legend()
    plt.savefig("algorithm_comparison.png")
    plt.show()

    # 3. Print policies
    print_policy(Q_ql, label="Q-Learning Policy")
    print_policy(Q_sa, label="SARSA Policy")

    # 4. Compare Q-values near the hole
    # Grid indices: State 6 is typically the hole in a 4x4 setup
    near_hole = {2: "Above Hole", 5: "Left of Hole", 7: "Right of Hole", 10: "Below Hole"}
    
    print(f"{'State':<8} | {'Label':<15} | {'Max Q (QL)':<12} | {'Max Q (SA)':<12}")
    print("-" * 55)
    for s, label in near_hole.items():
        max_ql = np.max(Q_ql[s])
        max_sa = np.max(Q_sa[s])
        print(f"{s:<8} | {label:<15} | {max_ql:<12.4f} | {max_sa:<12.4f}")

    # 5. Print final averages
    print(f"\nFinal 50-ep average (QL): {np.mean(rewards_ql[-50:]):.2f}")
    print(f"Final 50-ep average (SA): {np.mean(rewards_sa[-50:]):.2f}")

    # pass



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
    # 1. Compute V as max over actions and reshape to 4x4
    V = np.max(Q, axis=1)
    V_grid = V.reshape((4, 4))

    # 2. Create figure and axis
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 3. Display the heatmap
    # RdYlGn (Red-Yellow-Green) is perfect for showing reward gradients
    im = ax.imshow(V_grid, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im)

    # 4. Loop over cells to add text labels
    for row in range(4):
        for col in range(4):
            s = row * 4 + col
            val = V_grid[row, col]
            
            # Determine text based on state type
            if s == GridWorld.GOAL:
                text = "G"
            elif s == GridWorld.HOLE:
                text = "H"
            else:
                text = f"{val:.1f}"
            
            # Add text to the center of each tile
            ax.text(col, row, text, ha="center", va="center", 
                    fontsize=12, fontweight="bold")

    # 5. Set tick labels and titles
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f"col {c}" for c in range(4)])
    ax.set_yticklabels([f"row {r}" for r in range(4)])
    ax.set_title(title)
    
    # 6. Finalize and save
    plt.tight_layout()
    plt.savefig("value_heatmap.png")
    plt.show()
    # pass



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
    env = GridWorld()
    state = env.reset()
    total_reward = 0.0
    done = False
    step = 0
    reached_goal = False

    while not done and step < max_steps:
        step += 1
        
        # 1. Choose action
        action = choose_action(state, Q, epsilon)
        
        # 2. Get info for printing before state changes
        row, col = env.state_to_coords(state)
        action_name = GridWorld.ACTION_NAMES[action]
        
        # 3. Take step
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # 4. Print current step info
        # Format: Step  1 | State:  0 (row 0, col 0) | Action: DOWN  | Reward: -0.1
        print(f"Step {step:>2} | State: {state:>2} (row {row}, col {col}) | "
              f"Action: {action_name:<5} | Reward: {reward:.1f}")
        
        # 5. Update state
        state = next_state
        
        # Check if goal reached
        if state == GridWorld.GOAL:
            reached_goal = True

    # Final summary
    print(f"\nEpisode finished in {step} steps. "
          f"Total reward: {total_reward:.2f}. "
          f"Reached goal: {reached_goal}")




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
