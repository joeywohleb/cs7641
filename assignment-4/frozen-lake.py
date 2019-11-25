import gym
import numpy as np
from hiive.mdptoolbox.mdp import ValueIteration, QLearning, PolicyIteration
import matplotlib
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

Gamma = 0.99

env.reset()

P = np.zeros((4, 16, 16))
R = np.zeros((16, 4))
# prepare gym for mdptoolbox
for state in env.env.P:
    for action in env.env.P[state]:
        for option in env.env.P[state][action]:
            P[action][state][option[1]] += option[0]
            R[state][action] += option[2]


VI = ValueIteration(P, R, Gamma, 0.1, 20000)

# run VI
VI.setVerbose()
VI.run()
print('VI')
print(VI.iter)
print(VI.time)
print(VI.run_stats[-1:])

iterations = np.zeros(len(VI.run_stats))
reward = np.zeros(len(VI.run_stats))
i = 0
for stat in VI.run_stats:
    iterations[i] = stat['Iteration']
    reward[i] = stat['Reward']
    i += 1

fig, ax = plt.subplots()
ax.plot(iterations, reward)

ax.set(xlabel='Iterations', ylabel='Reward',
       title='Frozen Lake Value Iteration')
ax.grid()

fig.savefig("frozen-lake.vi.png")

PI = PolicyIteration(P, R, Gamma, None, 20000)

# run PI
PI.setVerbose()
PI.run()
print('PI')
print(PI.iter)
print(PI.time)
print(PI.run_stats[-1:])

iterations = np.zeros(len(PI.run_stats))
reward = np.zeros(len(PI.run_stats))
i = 0
for stat in PI.run_stats:
    iterations[i] = stat['Iteration']
    reward[i] = stat['Reward']
    i += 1

fig, ax = plt.subplots()
ax.plot(iterations, reward)

ax.set(xlabel='Iterations', ylabel='Reward',
       title='Frozen Lake Policy Iteration')
ax.grid()

fig.savefig("frozen-lake.pi.png")

values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.99]
resultRewards = [None] * len(values)
resultIterations = [None] * len(values)
i = 0
for v in values:
    QL = QLearning(P, R, Gamma, n_iter=100000,
                   epsilon=1, epsilon_decay=v, epsilon_min=0.1)
    # run QL
    QL.setVerbose()
    QL.run()
    print('QL')
    print(QL.time)
    print(QL.run_stats[-1:])

    resultIterations[i] = np.zeros(len(QL.run_stats))
    resultRewards[i] = np.zeros(len(QL.run_stats))
    j = 0
    sum = 0
    for stat in QL.run_stats:
        sum += stat['Reward']
        resultIterations[i][j] = stat['Iteration']
        resultRewards[i][j] = sum
        j += 1

    i += 1

fig, ax = plt.subplots()

for i in range(len(values)):
    ax.plot(resultIterations[i], resultRewards[i], label=values[i])

ax.set(xlabel='Iterations', ylabel='Accumulated Reward',
       title='Frozen Lake Q-Learning')
ax.grid()
ax.legend()
fig.savefig("frozen-lake.ql.decay.png")


QL = QLearning(P, R, Gamma, n_iter=1000000, epsilon_decay=0.1)
# run QL
QL.setVerbose()
QL.run()
print('QL')
print(QL.time)
print(QL.run_stats[-1:])


iterations = np.zeros(len(QL.run_stats))
reward = np.zeros(len(QL.run_stats))
i = 0
sum = 0
for stat in QL.run_stats:
    sum += stat['Reward']
    iterations[i] = stat['Iteration']
    reward[i] = sum
    i += 1

fig, ax = plt.subplots()
ax.plot(iterations, reward)

ax.set(xlabel='Iterations', ylabel='Accumulated Reward',
       title='Frozen Lake Q-Learning')
ax.grid()
fig.savefig("frozen-lake.ql.png")

print(QL.policy == VI.policy)
print(VI.policy == PI.policy)
print(VI.policy)
print(PI.policy)
print(QL.policy)
