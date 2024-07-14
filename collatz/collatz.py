import matplotlib.pyplot as plt
import numpy as np
import math
import time
import argparse

def collatz(n):
    if n%2==0:
        return n//2
    else:
        return 3*n+1

def do_collatz(n):
    sols=[n]
    while True:
        n = collatz(n)
        sols.append(n)
        if n==1:
            return sols

def plot_collatz(sols):
    plt.plot(sols, color='blue')
    plt.plot(0,sols[0], 'o', color='green')
    for i in range(1, len(sols)):
        plt.plot(i,sols[i], 'o', color='red')
    # yint=range(min(sols), max(sols)+1,2)
    xint=range(0,len(sols))
    plt.xticks(xint)
    # plt.yticks(yint)
    plt.xlabel('Step Number')
    plt.ylabel('Number')
    plt.title(f'Collatz Numbers for {sols[0]}')
    plt.show()
    return None

def main(n):
    print(f'Computing Collatz Numbers for {n}')
    start = time.time()
    sols = do_collatz(n)
    end = time.time()
    print(f'Time Taken: {end-start}')
    plot_collatz(sols)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and Plot Collatz Numbers')
    parser.add_argument('-n','--number', help='inpute number to compute collatz for', default=4, required=True)
    args = parser.parse_args()
    num = int(args.number)
    main(num)
