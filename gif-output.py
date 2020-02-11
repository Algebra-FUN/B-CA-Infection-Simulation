'''
gif-output.py
Brownian-CA Infection Simulation's GIF output script
Copyright 2020 by Algebra-FUN
ALL RIGHTS RESERVED.
'''

import imageio

days = 100

frames = []
for i in range(days):
    frames.append(imageio.imread(r'./temp/day{}.png'.format(i)))
imageio.mimsave(r'./img/{}days.gif'.format(days), frames, 'GIF', duration = 0.1)