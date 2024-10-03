import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
import os

#palette = ["#c7f9cc", "#80ed99", "#57cc99", "#38a3a5", "#22577a"]
#palette = ["#A3F5AB", "#70EB8D", "#57cc99", "#38a3a5", "#22577a"]
#palette = ["#ef476f", "#f78c6b", "#ffd166", "#06d6a0", "#118ab2"]
palette = ['#FFC533', '#f48c06', '#DD5703', '#d00000', '#6A040F']
cmap = LinearSegmentedColormap.from_list("paper", palette)

IMG_PATH = 'img/'
os.makedirs(IMG_PATH, exist_ok=True)

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}
"""
K_2_LAYER_2_CLUSTER = {
    1: {0: 7, 1: 7, 2: 7, 3: 7, 4: 7, 5: 7, 6: 7, 7: 7, 8: 7, 9: 7, 10: 7, 11: None},
    2: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4, 11: None},
    3: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 3, 6: 3, 7: 3, 8: 6, 9: 6, 10: 6, 11: None},
    4: {0: 0, 1: 0, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 6, 9: 6, 10: 6, 11: None},
    5: {0: 0, 1: 0, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 5, 9: 5, 10: None, 11: None},
}

CLUSTER_IDS = {
    0: [0, 1],
    1: [0, 1, 2, 3, 4],
    2: [2, 3, 4],
    3: [5, 6, 7],
    4: [5, 6, 7, 8, 9, 10],
    5: [8, 9],
    6: [8, 9, 10],
    7: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
"""

K_2_LAYER_2_CLUSTER = {
    1: {0: 8, 1: 8, 2: 8, 3: 8, 4: 8, 5: 8, 6: 8, 7: 8, 8: 8, 9: 8, 10: 8, 11: None},
    2: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1, 10: 1, 11: None},
    3: {0: 2, 1: 2, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 1, 8: 1, 9: 1, 10: 1, 11: None},
    4: {0: 2, 1: 2, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: None},
    5: {0: 2, 1: 2, 2: 2, 3: 6, 4: 6, 5: 7, 6: 7, 7: 4, 8: 4, 9: 5, 10: 5, 11: None},
}

CLUSTER_IDS = {
    0: [0, 1, 2, 3, 4, 5, 6],
    1: [7, 8, 9, 10],
    2: [0, 1, 2],
    3: [3, 4, 5, 6],
    4: [7, 8],
    5: [9, 10],
    6: [3, 4],
    7: [5, 6],
    8: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

