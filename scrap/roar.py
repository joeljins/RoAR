import random
import numpy as np

d = 3   #dimensions
x = [random.uniform(0, 10) for _ in range(d)]
x_prime = [random.uniform(0, 10) for _ in range(d)]
theta = [random.uniform(0, 10) for _ in range(d)]
theta_prime = [0] * d 
l = random.uniform(0, 3)
c = random.uniform(0, 3)
alpha = random.uniform(0, 3)

active = list(range(d))

for i in active[:]:
    if x[i] != 0:
        theta_prime[i] = theta[i] + alpha * np.sign(x[i])
    else:
        if abs(theta[i]) > alpha:
            theta_prime[i] = theta[i] + alpha * np.sign(theta[i])
        else:
            active.remove(i)
'''
while True:
    temp = [theta_prime[i] for i in active]
    i = temp.index(max(temp))
    delta = '
    '''