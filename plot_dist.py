"""
Rec-sys Embedding Rec-sys prototyping

__author__:
    charles@qileap

"""
import random
import numpy as np
def sample_gen(n, forbid):
    state = dict()
    track = dict()
    for (i, o) in enumerate(forbid):
        x = track.get(o, o)
        t = state.get(n-i-1, n-i-1)
        state[x] = t
        track[t] = x
        state.pop(n-i-1, None)
        track.pop(o, None)
    del track
    for remaining in range(n-len(forbid), 0, -1):
        i = random.randrange(remaining)
        yield state.get(i, i)
        state[i] = state.get(remaining - 1, remaining - 1)
        state.pop(remaining - 1, None)
import seaborn as sns
# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference
result = []
for i in range(1000):
    # result.append(random.gauss(5, 10))
    result.append(random.expovariate(4))
    # result.append(random.random())
# sns.distplot(result)
def project(l1, low, high):
    OldMax = max(l1)
    OldMin = min(l1)
    NewMax = high
    NewMin = low
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)
    NewValue = [(((x - OldMin) * NewRange) / OldRange) + NewMin for x in l1]
    return NewValue

projected = softmax(result)
projected = project(result, 0, 1)
print(max(projected), min(projected))
# print(project)
projected = sorted(projected,reverse=True)
print(projected)
projected = [int(x * 400) for x in projected]
print(projected)
# sns.plt.plot(project)
sns.distplot(projected)
sns.plt.show()
exit()