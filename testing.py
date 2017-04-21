rewards =  {1: 90, 10: 5}
times = sorted(rewards.keys())
t = next(val for x, val in enumerate(times) if val >= 10)
print t