from sae_lens import HookedSAETransformer
from tqdm import tqdm
import json
import os
import pandas as pd

os.environ["HF_HOME"] = "/workspace/huggingface"

from circuit import IOICircuit, IOIPrompt

device = "cuda"

def logits_diff(logits, correct_answer, incorrect_answer=None):
    correct_index = model.to_single_token(correct_answer)
    if incorrect_answer is None:
        return logits.cpu().numpy()[0, -1, correct_index]
    else:
        incorrect_index = model.to_single_token(incorrect_answer)
        return logits.cpu().numpy()[0, -1, correct_index] - logits.cpu().numpy()[0, -1, incorrect_index]

def logits_score(clean_logits, patched_logits, corr_logits, correct_answer, incorrect_answer=None):
    clean_score = logits_diff(clean_logits, correct_answer, incorrect_answer)
    patched_score = logits_diff(patched_logits, correct_answer, incorrect_answer)
    corr_score = logits_diff(corr_logits, correct_answer, incorrect_answer)

    score = (patched_score - clean_score) / (corr_score - clean_score)
    return score

# Load task
with open('tasks/ioi/cfg.json') as f:
    cfg = json.load(f)

with open('tasks/ioi/prompts.json') as f:
    prompts = json.load(f)

with open('tasks/ioi/names.json') as f:
    names = json.load(f)


# Load model and SAEs
model: HookedSAETransformer = HookedSAETransformer.from_pretrained("gpt2").to(device)
model.eval()

ioi_circuit = IOICircuit(
    model=model,
    cfg=cfg,
    prompts=prompts,
    names=names)

ioi_circuit.load_saes('z')
ioi_circuit.load_saes('resid_pre')

ioi_circuit.compute_supervised_dictionary()

all_nodes_labels = ['bNMH-q', 'bNMH-qk'] #['IH+DTH-z', 'SIH-v', 'SIH-z', 'bNMH-q', 'bNMH-qk', 'bNMH-z']
all_nodes = [['bNMH.q'], ['bNMH.qk']] #[['IH.z', 'DTH.z'], ['SIH.v'], ['SIH.z'], ['bNMH.q'], ['bNMH.qk'], ['bNMH.z']]

for node_names, node_label in zip(all_nodes, all_nodes_labels):
    
    scores_df = {
        'clean_ld': [],
        'supervised_full_ld': [],
        'sae_full_ld': [],
        'supervised_average_ld': [],
        'sae_average_ld': [],
        'ablation_ld': []
    }

    for idx in tqdm(range(512)):

        example = IOIPrompt(prompts[idx])
        example.tokenize(model)
        io = example.get_variable('IO')
        s = example.get_variable('S')

        ### SUPERVISED DICTIONARY - FULL
        clean_logits, patched_logits = ioi_circuit.run_with_reconstruction(
            example, 
            node_names=node_names,
            method='supervised',
            reconstruction='full',
            verbose=False
            )

        scores_df['clean_ld'].append(logits_diff(clean_logits, io, s).item())
        scores_df['supervised_full_ld'].append(logits_diff(patched_logits, io, s).item())

        ### SAE FEATURES - FULL
        clean_logits, patched_logits = ioi_circuit.run_with_reconstruction(
            example, 
            node_names=node_names,
            method='sae',
            reconstruction='full',
            verbose=False
            )
    
        scores_df['sae_full_ld'].append(logits_diff(patched_logits, io, s).item())

        ### SUPERVISED DICTIONARY - AVERAGE
        clean_logits, patched_logits = ioi_circuit.run_with_reconstruction(
            example, 
            node_names=node_names,
            method='supervised',
            reconstruction='average',
            verbose=False
            )

        scores_df['supervised_average_ld'].append(logits_diff(patched_logits, io, s).item())

        ### SAE FEATURES - AVERAGE
        clean_logits, patched_logits = ioi_circuit.run_with_reconstruction(
            example, 
            node_names=node_names,
            method='sae',
            reconstruction='average',
            verbose=False
            )
    
        scores_df['sae_average_ld'].append(logits_diff(patched_logits, io, s).item())

        ### ABLATION
        clean_logits, patched_logits = ioi_circuit.run_with_reconstruction(
            example, 
            node_names=node_names,
            method='ablation',
            reconstruction='average',
            verbose=False
            )
    
        scores_df['ablation_ld'].append(logits_diff(patched_logits, io, s).item())

    scores_df = pd.DataFrame(scores_df)
    scores_df.to_json(f'tasks/ioi/sn-scores/{node_label}.json')
