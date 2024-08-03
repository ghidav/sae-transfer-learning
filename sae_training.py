import torch
import os

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens import SAETrainingRunner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_HOME'] = '/workspace/huggingface'

training_tokens = 2_000_000_000
batch_size = 4096

total_training_steps = training_tokens // batch_size

component = "hook_resid_post"

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

for i in range(0, 6):
    cfg = LanguageModelSAERunnerConfig(
        
        # Data Generating Function (Model + Training Distibuion)
        model_name = "EleutherAI/pythia-70m-deduped",
        hook_name = f"blocks.{i}.{component}",
        hook_layer = i,
        dataset_path = "NeelNanda/pile-small-tokenized-2b",
        is_dataset_tokenized = True,
        context_size = 1024,

        # SAE Parameters
        architecture = "standard",
        d_in = 512,
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
        store_batch_size_prompts = 16,
        train_batch_size_tokens = batch_size,
        normalize_activations = (
            "none"  # none, expected_average_only_in (Anthropic April Update), constant_norm_rescale (Anthropic Feb Update)
        ),

        # Misc
        device = "cuda",
        seed = 42,
        dtype = "float32",
        prepend_bos = False,

        # Training Parameters

        ## Adam
        adam_beta1 = 0,
        adam_beta2 = 0.999,

        ## Loss Function
        mse_loss_normalization = None,
        l1_coefficient = 1,
        lp_norm = 1,
        scale_sparsity_penalty_by_decoder_norm = False,
        l1_warm_up_steps = l1_warm_up_steps,

        ## Learning Rate Schedule
        lr = 7e-5,
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

        # Misc
        resume = False,
        n_checkpoints = 0,
        checkpoint_path = "checkpoints",
        verbose = True
    )
    
    sparse_autoencoder = SAETrainingRunner(cfg).run()