import sys, os
import numpy as np
from einops import rearrange

import jax
import jax.numpy as jnp
from jax import random

from utils import scatter_3d

def update(x, key, dt=0.05, G=1):
    d = dist(x)
    norm = jnp.linalg.norm(d, axis=2) + jnp.eye(d.shape[0], d.shape[1])
    norm = jnp.tile(norm[..., None], d.shape[-1])
    d_norm = (G / x.shape[0]) * (d / norm)
    f = jnp.sum(d_norm, axis=1)
    f += 0.25*jnp.mean(x, axis=0)

    x -= f*dt
    x += 0.05*random.normal(key, f.shape)
    return x

def dist(x):
    x_ = x[None, ...]
    y1 = jnp.vstack([x_]*x.shape[0])
    y2 = rearrange(y1, 'b1 b2 d -> b2 b1 d')
    return y2 - y1

if __name__ == '__main__':
    d = 3
    n = 64
    #key = random.PRNGKey(np.random.randint(0, 100000)) 
    key = random.PRNGKey(2) 

    # generate random positions from 2 gaussians
    key, subkey = random.split(key)
    m = 2*random.normal(subkey, (2,d))

    key, subkey = random.split(key)
    x = random.normal(subkey, (n,d))

    x = x.at[:n//2].set(x[:n//2] + m[0])
    x = x.at[n//2:].set(x[n//2:] + m[1])

    x_all = [x]
    frames = 200
    # make random positions, and update key
    for i in range(frames):
        key, subkey = random.split(key)
        x = update(x, subkey)
        x_all.append(x)

    # convert to jax
    x_all = jnp.array(x_all)

    # make animation
    scatter_3d(x_all, frames)
