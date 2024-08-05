import torch
import os

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens import SAETrainingRunner
import argparse

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

argparser = argparse.ArgumentParser()
argparser.add_argument("-l", "--layer", type=int)
argparser.add_argument("-c", "--component", type=str)
argparser.add_argument("-l1", "--l1_coef", type=float)
args = argparser.parse_args()

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_HOME'] = '/workspace/huggingface'

training_tokens = 1_000_000_000
batch_size = 4096

total_training_steps = training_tokens // batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

cfg = LanguageModelSAERunnerConfig(
    
    # Data Generating Function (Model + Training Distibuion)
    model_name = "pythia-160m-deduped",
    hook_name = f"blocks.{args.layer}.{args.component}",
    hook_layer = args.layer,
    dataset_path = "NeelNanda/pile-small-tokenized-2b",
    is_dataset_tokenized = True,
    context_size = 1024,
    streaming=False,

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
    training_tokens = training_tokens,
    store_batch_size_prompts = 8,
    train_batch_size_tokens = batch_size,
    normalize_activations = (
        "none"  # none, expected_average_only_in (Anthropic April Update), constant_norm_rescale (Anthropic Feb Update)
    ),

    # Misc
    device = device,
    seed = 42,
    dtype = "float32",
    prepend_bos = False,

    # Training Parameters

    ## Adam
    adam_beta1 = 0,
    adam_beta2 = 0.999,

    ## Loss Function
    mse_loss_normalization = None,
    l1_coefficient = args.l1_coef,
    lp_norm = 1,
    scale_sparsity_penalty_by_decoder_norm = False,
    l1_warm_up_steps = l1_warm_up_steps,

    ## Learning Rate Schedule
    lr = 3e-5,
    lr_scheduler_name = (
        "constant"  # constant, cosineannealing, cosineannealingwarmrestarts
    ),
    lr_warm_up_steps = lr_warm_up_steps,
    lr_end = None,  # only used for cosine annealing, default is lr / 10,
    lr_decay_steps = lr_decay_steps,

    # Resampling protocol args
    use_ghost_grads = False,  # want to change this to true on some timeline.,
    feature_sampling_window = 2000,
    dead_feature_window = 1000,  # unless this window is larger feature sampling,,

    dead_feature_threshold = 1e-6,

    # Evals
    n_eval_batches = 10,
    eval_batch_size_prompts = None,  # useful if evals cause OOM,

    # WANDB
    log_to_wandb = True,
    log_activations_store_to_wandb = False,
    log_optimizer_state_to_wandb = False,
    wandb_project = "sae-transfer-learning",
    wandb_log_frequency = 30,
    eval_every_n_wandb_logs = 100,
    run_name = f"L{args.layer}_{args.component}_L1_{str(args.l1_coef).replace('.', '_')}",

    # Misc
    resume = False,
    n_checkpoints = 10,
    checkpoint_path = "checkpoints",
    verbose = True
)

sparse_autoencoder = SAETrainingRunner(cfg).run()