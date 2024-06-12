import torch
import pandas as pd
import numpy as np
from sae_lens import LMSparseAutoencoderSessionloader
from huggingface_hub import snapshot_download
import os

from transformer_lens import utils
from functools import partial

os.environ['HF_HOME'] = '/workspace/huggingface'

device = "cuda" if torch.cuda.is_available() else "cpu"

layer = 8 # pick a layer you want.
REPO_ID = "ghidav/pythia-70m-deduped-sae"
FILENAME = f"test/blocks.4.hook_resid_pre"

path = snapshot_download(repo_id=REPO_ID)
print(path)
model, sparse_autoencoder, activation_store = LMSparseAutoencoderSessionloader.load_pretrained_sae(
    path=os.path.join(path, FILENAME), device=device
)
sparse_autoencoder.eval()

# L0 test
with torch.no_grad():
    batch_tokens = activation_store.get_batch_tokens()
    _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

    # Use the SAE
    sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
        cache[sparse_autoencoder.cfg.hook_point]
    )

    # save some room
    del cache

    # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
    l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
    print("Average l0", np.round(l0.mean().item(), 3))

# Reconstruction
def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

print("Original: ",
      np.round(model(
          batch_tokens, 
          return_type="loss"
      ).item(), 3),
)

print(
    "Reconstruction: ",
    np.round(model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[
            (
                sparse_autoencoder.cfg.hook_point,
                partial(reconstr_hook, sae_out=sae_out),
            )
        ],
        return_type="loss",
    ).item(), 3),
)

print(
    "Zero-ablation: ",
    np.round(model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(sparse_autoencoder.cfg.hook_point, zero_abl_hook)],
    ).item(), 3),
)