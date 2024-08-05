from sae_lens.evals import run_evals, EvalConfig
from sae_lens import SAE, ActivationsStore, HookedSAETransformer
from sae_lens.config import LanguageModelSAERunnerConfig
import argparse
from sae_lens.evals import run_evals, EvalConfig
from sae_lens import SAE, ActivationsStore, HookedSAETransformer
from sae_lens.config import LanguageModelSAERunnerConfig
import argparse
import torch
import pandas as pd
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--component", type=str)
args = parser.parse_args()


default_cfg = LanguageModelSAERunnerConfig(
    
    # Data Generating Function (Model + Training Distibuion)
    model_name = "pythia-160m-deduped",
    hook_name = None,
    hook_layer = None,
    dataset_path = "NeelNanda/pile-small-tokenized-2b",
    is_dataset_tokenized = True,
    context_size = 1024,
    streaming=True,

    # SAE Parameters
    architecture = "jumprelu",
    d_in = 768,
    d_sae = None,
    b_dec_init_method = "zeros",
    expansion_factor = 8,
    activation_fn = "relu",  # relu, tanh-relu, topk
    normalize_sae_decoder = True,
    from_pretrained_path = None,
    apply_b_dec_to_input = False,

    # Activation Store Parameters
    n_batches_in_buffer = 128,

    # Misc
    device = device,
    seed = 42,
    dtype = "float32",
    prepend_bos = False
)

eval_cfg = EvalConfig(
    batch_size_prompts = 8,

    # Reconstruction metrics
    n_eval_reconstruction_batches = 32,
    compute_kl = True,
    compute_ce_loss = True,

    # Sparsity and variance metrics
    n_eval_sparsity_variance_batches = 1,
    compute_l2_norms = True,
    compute_sparsity_metrics = True,
    compute_variance_metrics = False,
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

# Load model
model = HookedSAETransformer.from_pretrained("pythia-160m-deduped").to(device)

layers = model.cfg.n_layers
all_metrics = []

for i in tqdm(range(layers)):
    # Set activation store
    cfg = update_cfg(i, hook_name)
    activations_store = ActivationsStore.from_config(
        model,
        cfg
    )
    for j in range(layers):
        try:
            # Load SAE
            SAE_PATH = f"checkpoints/L{j}"
            sae = SAE.load_from_pretrained(SAE_PATH).to(device)

            metrics = run_evals(sae, activations_store, model, eval_cfg)
            metrics = {k.split('/')[-1]: v for k, v in metrics.items()}
            print(f"L{j} SAE on L{i} activations. C/E: {metrics['ce_loss_score']:.3f}")
            all_metrics.append(pd.Series(metrics, name=f"{i}-{j}"))
        except Exception as e:
            print(f"Failed to load L{j} SAE.", e)
            continue

all_metrics = pd.concat(all_metrics, axis=1).T
all_metrics.to_csv(f"eval/{component}.csv")