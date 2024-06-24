import torch

def feature_patching_hook(x, hook, pos, head, f_in, f_out):
    x[:, pos, head] = x[:, pos, head] + f_in[None, pos, head] - f_out[None, pos, head]
    return x

def patching_hook(x, hook, pos, patch, head=None):
    if head is None:
        x[:, pos] = patch
    else:
        x[:, pos, head] = patch
    return x

def editing_hook(x, hook, pos, edit, head=None):
    if head is None:
        x[:, pos] = x[:, pos] + edit
    else:
        x[:, pos, head] = x[:, pos, head] + edit