import torch

def zero_abl_hook(x, hook, pos, head):
    x[:, pos, head] = 0
    return x

def patching_hook(x, hook, pos, head, corr):
    x[:, pos, head] = corr[:, pos, head]
    return x

def feature_patching_hook(x, hook, pos, head, f_in, f_out):
    x[:, pos, head] = x[:, pos, head] + f_in[None, pos, head] - f_out[None, pos, head]
    return x

def feature_z_patching_hook(x, hook, pos, f_in):
    x[:, pos] = x[:, pos] + f_in[None]
    return x

def feature_qkv_patching_hook(x, hook, pos, head, edit):
    x[:, pos, head] = edit
    return x