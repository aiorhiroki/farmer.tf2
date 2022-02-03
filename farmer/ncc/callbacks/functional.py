import numpy as np

def linear(param, epoch, delta=0.01, min=0.01, max=None):
    updated = True
    return updated, np.clip(param + delta, min, max)

def step_delta(param, epoch, delta=0.01, step=1, min=0.01, max=None):
    updated = False
    if (epoch + 1) % step == 0:
        updated = True
        return updated, np.clip(param + delta, min, max)
    else:
        return updated, param

