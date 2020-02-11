'''
B-CA-I Simulation.py
Brownian-CA Infection Simulation
Copyright 2020 by Algebra-FUN
ALL RIGHTS RESERVED.
'''

from random import randint
import numpy as np
from numpy.random import random
import pandas as pd
from matplotlib import pyplot as plt


def init_people_geo(n): return random(n)*10

remove_rate = .01
infect_rate = .5
cmap = list('grk')


class BrownianInfection:
    def __init__(self, N, D, v, plot):
        self.N = N
        self.v = v
        self.D = D
        self.plot = plot
        self.__init_people()
        self.__init_history()

    def __init_history(self):
        self.history = pd.DataFrame({'S': [self.N-1], 'I': [1], 'R': [0]})

    def __init_people(self):
        def init_people_geo(): return random(self.N)*self.D
        # init people
        self.people = pd.DataFrame({
            'x': init_people_geo(),
            'y': init_people_geo(),
            'status': np.zeros(self.N, dtype=int)
        })
        # init the first infector
        self.people.loc[randint(0, self.N-1), 'status'] = 1

    def display(self):
        self.display_position()
        self.display_history()

    def display_position(self):
        ax = self.plot(0)
        ax.cla()

        def colors():
            def color_map(code): return cmap[int(code)]
            return list(map(color_map, self.people['status']))

        ax.scatter(self.people['x'], self.people['y'],
                   c=colors())
        ax.set_title('v={},D={}'.format(self.v, self.D))

    def display_history(self):
        ax = self.plot(1)
        ax.cla()
        days = range(len(self.history))
        for i, label in enumerate(list('SIR')):
            ax.plot(days, self.history[label], cmap[i], label=label)
        ax.legend()
        ax.set_xlabel('day')
        ax.set_ylabel('count')

    def daily(self):    
        self.move()
        self.infect()
        self.remove()
        self.summerize()

    def move(self):
        def random_sign(i): return np.sign(random(i) - .5)
        xs = random(self.N) * random_sign(self.N)
        ys = (1 - xs ** 2) ** .5 * random_sign(self.N)
        dps = pd.DataFrame({'x': xs*self.v, 'y': ys*self.v})
        self.people.loc[:, ['x', 'y']] += dps

    def infect(self):
        infectors = self.people.query('status == 1')
        infected = len(infectors)
        for i in range(infected):
            ordinarys = self.people.query('status == 0')
            xs, ys = ordinarys['x'], ordinarys['y']
            x, y = infectors.iloc[i, 0], infectors.iloc[i, 1]
            dxs, dys = xs-x, ys-y
            ds = (dxs**2+dys**2)
            ps = np.exp(-infect_rate*ds)
            Ss = self.people['status'] == 0
            infectment = [1 if v < min(
                p, .8) and p > 0.1 else 0 for v, p in zip(random(len(Ss)), ps)]
            self.people.loc[Ss, 'status'] += infectment

    def remove(self):
        infected = len(self.people.query('status == 1'))
        removements = [
            1 if i < remove_rate else 0 for i in random(infected)]
        self.people.loc[self.people['status'] == 1, 'status'] += removements

    def summerize(self):
        dic = {}
        for i, key in enumerate(list('SIR')):
            dic[key] = [len(self.people.query('status == {}'.format(i)))]
        summary = pd.DataFrame(dic)
        self.history = self.history.append(summary)


def plots(i): return lambda k: axs[i][k]


bi1 = BrownianInfection(N=100, v=2, D=10, plot=plots(0))
bi2 = BrownianInfection(N=100, v=1, D=20, plot=plots(1))
bi3 = BrownianInfection(N=100, v=0, D=20, plot=plots(2))
bis = [bi1, bi2, bi3]

groups = len(bis)
fig, axs = plt.subplots(groups, 2, figsize=(8, 12))

days = 100
plt.subplots_adjust(wspace=.4, hspace=.6)

plt.ion()

for i in range(days):
    plt.cla()
    for bi in bis:
        bi.display()
        bi.daily()
    plt.pause(.2)
    plt.savefig(r'./temp/day{}.png'.format(i))

plt.ioff()
plt.show()
