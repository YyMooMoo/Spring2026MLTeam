# gridworld.py
# DO NOT MODIFY THIS FILE.
# This is the environment. Your agent code lives in agent.py.

import numpy as np


class GridWorld:
    """
    4x4 GridWorld environment.

    States:  integer = row * 4 + col  (0 to 15)
    Actions: 0=UP  1=DOWN  2=LEFT  3=RIGHT

    Layout:
        S  .  .  .    (row 0)
        .  .  H  .    (row 1)
        .  .  .  .    (row 2)
        .  .  .  G    (row 3)

    S = start  (state  0)
    G = goal   (state 15)  reward +10,  episode ends
    H = hole   (state  6)  reward -10,  episode ends
    . = empty             reward -0.1

    Moving into a wall keeps the agent in place.
    """

    NSTATES  = 16
    NACTIONS = 4
    START    = 0
    GOAL     = 15
    HOLE     = 6
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def __init__(self):
        self.state = self.START

    def reset(self):
        """Reset agent to start state. Returns initial state (int)."""
        self.state = self.START
        return self.state

    def step(self, action):
        """
        Take one step in the environment.

        Args:
            action (int): 0=UP  1=DOWN  2=LEFT  3=RIGHT

        Returns:
            next_state (int):   state after the action
            reward     (float): reward received
            done       (bool):  True if the episode has ended
        """
        row, col = self.state // 4, self.state % 4

        if   action == 0: row = max(row - 1, 0)
        elif action == 1: row = min(row + 1, 3)
        elif action == 2: col = max(col - 1, 0)
        elif action == 3: col = min(col + 1, 3)

        self.state = row * 4 + col

        if   self.state == self.GOAL: return self.state, +10.0, True
        elif self.state == self.HOLE: return self.state, -10.0, True
        else:                         return self.state,  -0.1, False

    def render(self, Q=None):
        """
        Print a text rendering of the grid.
        If Q is provided, show the greedy action arrow at each non-terminal cell.
        """
        symbols = {0: "^", 1: "v", 2: "<", 3: ">"}
        for row in range(4):
            line = ""
            for col in range(4):
                s = row * 4 + col
                if   s == self.START: line += "S  "
                elif s == self.GOAL:  line += "G  "
                elif s == self.HOLE:  line += "H  "
                elif Q is not None:   line += symbols[int(np.argmax(Q[s]))] + "  "
                else:                 line += ".  "
            print(line)
        print()

    def state_to_coords(self, state):
        """Convert state integer to (row, col) tuple."""
        return state // 4, state % 4
