from hiive.mdptoolbox.mdp import ValueIteration, QLearning, PolicyIteration
from hiive.mdptoolbox.example import forest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

P, R = forest(2000)

compare_VI_QI_policy = []  # True or False
compare_VI_PI_policy = []

Gamma = 1
Epsilon = 0.0000000000000000000000000000000000000000000000000000000000000000000000000001
Max_Iterations = 200000

VI = ValueIteration(P, R, Gamma, Epsilon, Max_Iterations)

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
       title='Forest Value Iteration')
ax.grid()

fig.savefig("forest.vi.png")


Gamma = 0.99
PI = PolicyIteration(P, R, Gamma)

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
       title='Forest Policy Iteration')
ax.grid()

fig.savefig("forest.pi.png")

QL = QLearning(P, R, Gamma, n_iter=1000000, alpha_decay=0.1)
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

ax.set(xlabel='Iterations', ylabel='Accumlated Reward',
       title='Forest Q-Learning')
ax.grid()
fig.savefig("forest.ql.png")

print(QL.policy == VI.policy)
print(VI.policy == PI.policy)
print('VI')
print('cut down')
print(np.count_nonzero(VI.policy))
print('wait')
print(len(VI.policy) - np.count_nonzero(VI.policy))
print('PI')
print('cut down')
print(np.count_nonzero(PI.policy))
print('wait')
print(len(PI.policy) - np.count_nonzero(PI.policy))
print('QL')
print('cut down')
print(np.count_nonzero(QL.policy))
print('wait')
print(len(QL.policy) - np.count_nonzero(QL.policy))
