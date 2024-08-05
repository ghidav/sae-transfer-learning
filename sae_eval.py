from sae_lens.evals import run_evals, EvalConfig
from sae_lens import SAE, ActivationsStore, HookedSAETransformer
from sae_lens.config import LanguageModelSAERunnerConfig
import argparse
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("-sae", "--sae", type=str)
parser.add_argument("-act", "--act", type=str)
args = parser.parse_args()

sae_layer, sae_component = args.sae.split("c")
sae_layer = int(sae_layer[1:])

act_layer, act_component = args.act.split("c")
act_layer = int(act_layer[1:])

# SAE Config
cfg = LanguageModelSAERunnerConfig(
    
    # Data Generating Function (Model + Training Distibuion)
    model_name = "pythia-160m-deduped",
    hook_name = f"blocks.{act_layer}.{act_component}",
    hook_layer = act_layer,
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
    #training_tokens = 1_000_000_000,
    #store_batch_size_prompts = 8,
    #train_batch_size_tokens = batch_size,
    #normalize_activations = (
    #    "none"  # none, expected_average_only_in (Anthropic April Update), constant_norm_rescale (Anthropic Feb Update)
    #),

    # Misc
    device = device,
    seed = 42,
    dtype = "float32",
    prepend_bos = False,

    # Training Parameters

    ## Adam
    #adam_beta1 = 0,
    #adam_beta2 = 0.999,

    ## Loss Function
    #mse_loss_normalization = None,
    #l1_coefficient = args.l1_coef,
    #lp_norm = 1,
    #scale_sparsity_penalty_by_decoder_norm = False,
    #l1_warm_up_steps = l1_warm_up_steps,

    ## Learning Rate Schedule
    #lr = 3e-5,
    #lr_scheduler_name = (
    #    "constant"  # constant, cosineannealing, cosineannealingwarmrestarts
    #),
    #lr_warm_up_steps = lr_warm_up_steps,
    #lr_end = None,  # only used for cosine annealing, default is lr / 10,
    #lr_decay_steps = lr_decay_steps,

    # Resampling protocol args
    #use_ghost_grads = False,  # want to change this to true on some timeline.,
    #feature_sampling_window = 2000,
    #dead_feature_window = 1000,  # unless this window is larger feature sampling,,

    #dead_feature_threshold = 1e-6,

    # Evals
    #n_eval_batches = 10,
    #eval_batch_size_prompts = None,  # useful if evals cause OOM,

    # WANDB
    #log_to_wandb = False,
    #log_activations_store_to_wandb = False,
    #log_optimizer_state_to_wandb = False,
    #wandb_project = "sae-transfer-learning",
    #wandb_log_frequency = 30,
    #eval_every_n_wandb_logs = 100,
    #run_name = f"L{args.layer}_{args.component}_L1_{str(args.l1_coef).replace('.', '_')}",

    # Misc
    #resume = False,
    #n_checkpoints = 10,
    #checkpoint_path = "checkpoints",
    #verbose = True
)

# Eval Config
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

# Load SAE
if sae_component == "RES":
    sae_component = "rs-post"
elif sae_component == "MLP":
    sae_component = "mlp-out"
elif sae_component == "ATT":
    sae_component = "attn-z"

SAE_PATH = f"/Users/ghidav/.cache/huggingface/hub/models--mech-interp--pythia-160m-deduped-{sae_component}/snapshots/3b8e8bffff1cf13322769107ecf50ceb23c406ee/L{sae_layer}"
sae = SAE.load_from_pretrained(SAE_PATH).to(device)

# Load model
model = HookedSAETransformer.from_pretrained("pythia-160m-deduped").to(device)

# Load activations store
activations_store = ActivationsStore.from_config(
        model,
        cfg
    )

if __name__ == "__main__":
    run_evals(sae, activations_store, model, eval_cfg)

# python3 sae_eval.py -sae l1cRES -act l0cRES