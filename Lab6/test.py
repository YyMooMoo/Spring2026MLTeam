from gridworld import GridWorld
env = GridWorld()
state = env.reset()
print("Start state:", state)

next_state, reward, done = env.step(1)   # move DOWN
print("After DOWN:", next_state, reward, done)

env.render()
