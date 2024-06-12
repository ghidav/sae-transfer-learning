import torch
import os

from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.lm_runner import SAETrainingRunner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_HOME'] = '/workspace/huggingface'

total_training_steps = 30_000  # probably we should do more
batch_size = 2048
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training
for i in range(0, 6):
    cfg = LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distibuion)
        model_name="marco-molinari_pythia-70m-instruct",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
        hook_point=f"blocks.{i}.hook_resid_pre",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
        hook_point_layer=i,  # Only one layer in the model.
        d_in=512,  # the width of the mlp output.
        dataset_path="NeelNanda/pile-small-tokenized-2b",  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
        is_dataset_tokenized=True,
        streaming=True,  # we could pre-download the token dataset if it was small.
        # SAE Parameters
        mse_loss_normalization=None,  # We won't normalize the mse loss,
        expansion_factor=32,  # the width of the SAE. Larger will result in better stats but slower training. --> 32768
        b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
        apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        normalize_activations=True,
        # Training Parameters
        lr=5e-5,  # lower the better, we'll go fairly high to speed up the tutorial.
        adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
        adam_beta2=0.999,
        lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
        lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
        lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
        l1_coefficient=5,  # will control how sparse the feature activations are
        l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
        lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
        train_batch_size_tokens=batch_size,
        context_size=256,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
        # Activation Store Parameters
        n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
        training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
        store_batch_size_prompts=16,
        # Resampling protocol
        use_ghost_grads=False,  # we don't use ghost grads anymore.
        feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
        dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
        dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
        # WANDB
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project="pythia-70m-deduped-sae",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
        # Misc
        device=device,
        seed=77,
        n_checkpoints=1,
        checkpoint_path="sae-transfer-learning/checkpoints-third-tuned-run",
        dtype=torch.float32,
    )
    
    sparse_autoencoder = SAETrainingRunner(cfg).run()