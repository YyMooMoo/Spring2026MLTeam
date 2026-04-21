# MODIFY THIS FILE.
# Complete every section marked with TODO.
# Do not change any function signatures.

import numpy as np
from gridworld import GridWorld


# You will tune these in Parts C and D via run.py arguments.
# Do not hardcode these values inside your functions.
ALPHA         = 0.1    # learning rate
GAMMA         = 0.9    # discount factor
EPSILON_START = 1.0    # initial exploration rate
EPSILON_END   = 0.01   # minimum exploration rate
EPSILON_DECAY = 0.995  # multiplicative decay per episode
EPISODES      = 600


def choose_action(state, Q, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, GridWorld.NACTIONS)
    return int(np.argmax(Q[state]))

    """
    ε-greedy action selection.

    With probability epsilon, return a random action (explore).
    Otherwise return the action with the highest Q-value (exploit).

    Args:
        state   (int):        current state (0-15)
        Q       (np.ndarray): Q-table of shape (16, 4)
        epsilon (float):      exploration probability in [0, 1]

    Returns:
        action (int): chosen action — 0=UP  1=DOWN  2=LEFT  3=RIGHT

    Hints:
        np.random.random()                    → uniform float in [0, 1)
        np.argmax(Q[state])                   → index of max value
        np.random.randint(0, GridWorld.NACTIONS) → random action
    """
    # TODO
    # With probability epsilon: return a random action
    # Otherwise:                return argmax Q[state]
    pass


def update_Q_learning(Q, state, action, reward, next_state, alpha, gamma):
    td_target = reward + gamma * np.max(Q[next_state]) # reward + gamma * max Q[next_state]
    td_error = td_target - Q[state, action] # td_target - Q[state, action]
    Q[state, action] += alpha * td_error     # Q[state, action] += alpha * td_error

    """
    One Q-learning (off-policy TD) update. Modifies Q in place.

    Update rule:
        td_target        = reward + gamma * max_a' Q[next_state, a']
        td_error (delta) = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

    Args:
        Q          (np.ndarray): Q-table, shape (16, 4) — update IN PLACE
        state      (int)
        action     (int)
        reward     (float)
        next_state (int)
        alpha      (float): learning rate
        gamma      (float): discount factor

    Hints:
        np.max(Q[next_state])  → best Q-value available from next_state
    """
    pass


def update_SARSA(Q, state, action, reward, next_state, next_action, alpha, gamma):
    td_target = reward + gamma * Q[next_state, next_action]
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error   
    """
    One SARSA (on-policy TD) update. Modifies Q in place.

    Identical structure to Q-learning with ONE change:
    instead of max_a' Q[next_state], use Q[next_state, next_action]
    where next_action is the action the policy ACTUALLY chose next.

    Update rule:
        td_target        = reward + gamma * Q[next_state, next_action]
        td_error (delta) = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

    Args:
        Q           (np.ndarray): Q-table, shape (16, 4) — update IN PLACE
        state       (int)
        action      (int)
        reward      (float)
        next_state  (int)
        next_action (int): the action the policy will ACTUALLY take next
        alpha       (float)
        gamma       (float)
    """
    # TODO
    # reward + gamma * Q[next_state, next_action]
    # Q[state, action] += alpha * td_error
    pass


def train(
    algorithm     = "qlearning",
    episodes      = EPISODES,
    alpha         = ALPHA,
    gamma         = GAMMA,
    epsilon_start = EPSILON_START,
    epsilon_end   = EPSILON_END,
    epsilon_decay = EPSILON_DECAY,
):
    env         = GridWorld()
    Q           = np.zeros((GridWorld.NSTATES, GridWorld.NACTIONS))
    rewards_log = []
    epsilon_log = []
    epsilon     = epsilon_start

    for episode in range(episodes):
        state        = env.reset()
        total_reward = 0.0
        done         = False

        epsilon_log.append(epsilon)

        if algorithm == "sarsa":
            action = choose_action(state, Q, epsilon)
        else:
            action = None

        while not done:

            if algorithm == "qlearning":
                action = choose_action(state, Q, epsilon)
                next_state, reward, done = env.step(action)
                update_Q_learning(Q, state, action, reward, next_state, alpha, gamma)
                state = next_state
                total_reward += reward

            elif algorithm == "sarsa":
                next_state, reward, done = env.step(action)

                if done:
                    next_action = 0
                else:
                    next_action = choose_action(next_state, Q, epsilon)

                update_SARSA(Q, state, action, reward, next_state, next_action, alpha, gamma)
                action = next_action
                state = next_state
                total_reward += reward

        rewards_log.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return Q, rewards_log, epsilon_log
