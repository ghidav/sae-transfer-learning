import torch

def zero_abl_hook(x, hook, pos, head):
    x[:, pos, head] = 0
    return x

def patching_hook(x, hook, pos, corr, head=None):
    if head is None:
        x[:, pos] = corr[:, pos]
    else:
        x[:, pos, head] = corr[:, pos, head]
    return x

def feature_patching_hook(x, hook, pos, head, f_in, f_out):
    x[:, pos, head] = x[:, pos, head] + f_in[None, pos, head] - f_out[None, pos, head]
    return x

def feature_editing_hook(x, hook, pos, edit, head=None):
    if head is None:
        x[:, pos] = edit
    else:
        x[:, pos, head] = edit
    return x