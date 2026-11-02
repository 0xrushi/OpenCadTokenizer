"""Pure-Python reference-edit reward; no model or geometry execution."""
from functools import lru_cache


def command_cost(a, b):
    if a == b:
        return 0.
    x, y = a.split(','), b.split(',')
    if x[0] != y[0] or len(x) != len(y) or len(x) == 1:
        return 1.
    # Rotation entries use -1/0/1; other SEM numeric fields use 0..63.
    costs = []
    for i, (u, v) in enumerate(zip(x[1:], y[1:])):
        span = 2. if x[0] in ('add','cut','intersect') and 5 <= i < 14 else 63.
        costs.append(min(1., abs(int(u)-int(v))/span))
    return sum(costs)/len(costs)


@lru_cache(maxsize=10000)
def distance(source, target):
    a, b = source.split(), target.split()
    previous = list(map(float, range(len(b)+1)))
    for i, value in enumerate(a, 1):
        row = [float(i)]
        for j, other in enumerate(b, 1):
            row.append(min(previous[j]+1, row[j-1]+1, previous[j-1]+command_cost(value,other)))
        previous = row
    return previous[-1]


def reward(source, target, prediction, valid):
    if not valid or prediction is None:
        return -1.
    baseline = distance(source, target)
    if baseline <= 0:
        raise ValueError('GRPO prompt must require a nonzero reference edit')
    progress = max(-1., min(1., 1-distance(prediction,target)/baseline))
    return progress + (.25 if prediction == target else 0.)


def advantages(rewards):
    import math
    mean = sum(rewards)/len(rewards)
    std = math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards))
    return [(r-mean)/(std+1e-4) for r in rewards], std
