import urllib3, socket
from urllib3.connection import HTTPConnection

HTTPConnection.default_socket_options = HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000),
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000),
]

import argparse
import json
import os

import pandas as pd
import torch
from datasets import load_dataset
from sae_lens import SAE, ActivationsStore, HookedSAETransformer
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.evals import EvalConfig, run_evals
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--component", type=str, default="RES")
args = parser.parse_args()


default_cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name="pythia-160m-deduped",
    hook_name=None,
    hook_layer=None,
    dataset_path="NeelNanda/pile-small-tokenized-2b",
    is_dataset_tokenized=True,
    context_size=1024,
    streaming=True,
    # SAE Parameters
    architecture="jumprelu",
    d_in=768,
    d_sae=None,
    b_dec_init_method="zeros",
    expansion_factor=8,
    activation_fn="relu",  # relu, tanh-relu, topk
    normalize_sae_decoder=True,
    from_pretrained_path=None,
    apply_b_dec_to_input=False,
    # Activation Store Parameters
    n_batches_in_buffer=128,
    # Misc
    device=device,
    seed=42,
    dtype="float32",
    prepend_bos=False,
)

eval_cfg = EvalConfig(
    batch_size_prompts=8,
    # Reconstruction metrics
    n_eval_reconstruction_batches=128,
    compute_kl=True,
    compute_ce_loss=True,
    # Sparsity and variance metrics
    n_eval_sparsity_variance_batches=128,
    compute_l2_norms=True,
    compute_sparsity_metrics=True,
    compute_variance_metrics=True,
)


def update_cfg(act_layer, hook_name):
    default_cfg.hook_layer = act_layer
    default_cfg.hook_name = f"blocks.{act_layer}.{hook_name}"
    return default_cfg


# Load SAE
if args.component == "RES":
    component = "rs-post"
    hook_name = "hook_resid_post"
elif args.component == "MLP":
    component = "mlp-out"
    hook_name = "hook_mlp_out"
elif args.component == "ATT":
    component = "attn-z"
    hook_name = "attn.hook_z"
else:
    raise ValueError("Invalid component.")

# Create checkpoint mapping
ckpt_folder = "/root/sae-transfer-learning/saes/pythia-160m-deduped/forward"
ckpt_step = "500M"
mapping = {}
for _dir in os.listdir(ckpt_folder):
    try:
        cfg = json.load(open(os.path.join(ckpt_folder, _dir, ckpt_step, "cfg.json")))
        mapping[_dir] = f"L{cfg['hook_name'].split('.')[1]}"
    except FileNotFoundError:
        continue
inv_mapping = {v: k for k, v in mapping.items()}

# Load model
model = HookedSAETransformer.from_pretrained("pythia-160m-deduped").to(device)
checkpoints = ["100M", "200M", "300M", "400M", "500M"]

start_layer = 0
end_layer = model.cfg.n_layers - 1
ckpt_folder = f"/root/sae-transfer-learning/saes/pythia-160m-deduped/forward"

dataset = load_dataset("NeelNanda/pile-small-tokenized-2b", streaming=True, split="train")

# Set activation store
for ckpt_step in checkpoints:
    print(f"Checkpoint: {ckpt_step}")
    all_transfer_metrics = []
    for sae_idx in tqdm(range(start_layer, end_layer)):
        cfg = update_cfg(sae_idx, hook_name)
        activations_store = ActivationsStore.from_config(model, cfg, override_dataset=dataset)
        try:
            # Load SAE
            TRANSFER_SAE_PATH = os.path.join(ckpt_folder, inv_mapping[f"L{sae_idx+1}"], ckpt_step)
            sae = SAE.load_from_pretrained(TRANSFER_SAE_PATH).to(device)
            sae.cfg.hook_name = f"blocks.{sae_idx+1}.{hook_name}"
            sae.cfg.hook_layer = sae_idx + 1
            metrics = run_evals(sae, activations_store, model, eval_cfg)
            metrics = {k.split("/")[-1]: v for k, v in metrics.items()}
            print(f"L{sae_idx+1} SAE on L{sae_idx+1} activations. C/E: {metrics['ce_loss_score']:.3f}")
            all_transfer_metrics.append(pd.Series(metrics, name=f"{sae_idx+1}-{sae_idx}"))
        except Exception as e:
            print(f"Failed to load L{sae_idx} SAE.", e)
            continue
    all_transfer_metrics = pd.concat(all_transfer_metrics, axis=1).T
    all_transfer_metrics.to_csv(f"eval/{component}_transfer_forward_{ckpt_step}_all_mse.csv")
