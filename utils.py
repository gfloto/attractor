import sys, os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# used for fast mapltolib animation
class Simulation:
    def __init__(self, x):
        assert x.shape[2] == 3
        assert len(x.shape) == 3

        self.x = x
        self.i = 0
        self.graph = None

    def step(self, first=False):
        if self.i == self.x.shape[0] - 1: time.sleep(1)
        else: time.sleep(0.1)
        if not first: self.i = (self.i + 1) % self.x.shape[0]
        return self.x[self.i].T

    # function to update the graph
    def update_graph(self, n):
        p = self.step()
        self.graph._offsets3d = (p[0], p[1], p[2])

# 3d plot
def scatter_3d(x, lim=3.):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    sim = Simulation(x)
    p = sim.step(first=True)
    graph = ax.scatter(p[0], p[1], p[2])
    sim.graph = graph

    # animate
    ani = animation.FuncAnimation(fig, sim.update_graph, interval=40, blit=False)
    plt.show()


