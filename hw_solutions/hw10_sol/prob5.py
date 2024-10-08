import numpy as np
from tqdm import tqdm

np.random.seed(0)

p =  18/37
q = 0.55
K = 600
upper_bound = 200
lower_bound = 0
N = 3000


count = 0

for i in tqdm(range(N)):
    money = 100  # initialize
    total = 1  # initialize
    win = None  # initialize
    for i in range(K):
        u = np.random.uniform(low=0., high=1., size=(1))
        if u < p:
            money += 1
        else:
            money -= 1
            
        if money == upper_bound or money == lower_bound:
            if win == None:
                if money == upper_bound:
                    win = True
                else:
                    win = False
            
    if win == True:
        count += 1

print('approximated probability: ', count/N)

sum = 0

for i in tqdm(range(N)):
    money = 100  # initialize money
    total = 1  # initialize total
    win = None
    for i in range(K):
        u = np.random.uniform(low=0., high=1., size=(1))
        if u < q:
            money += 1
            total *= p/q
        else:
            money -= 1
            total *= (1-p)/(1-q)
        
        if money == upper_bound or money == lower_bound:
            if win == None:
                if money == upper_bound:
                    win = True
                else:
                    win = False
            
    if win == True:
        sum += total

print('approximated probability: ', sum/N)