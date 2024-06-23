import torch
import pandas as pd
from sae_lens import HookedSAETransformer
from tqdm import tqdm
import json

from circuit import IOICircuit
from tasks.ioi.dictionary import supervised_dictionary

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
with open('tasks/ioi/task.json') as f:
    task = json.load(f)

# Load model and SAEs
model: HookedSAETransformer = HookedSAETransformer.from_pretrained("gpt2").to(device)
model.eval()

ioi_circuit = IOICircuit(model, task)
ioi_circuit.load_saes('z')
ioi_circuit.load_saes('resid_pre')

# Compute supervised dictionary
bs = 64
prompts = [p['prompt'] for p in task['prompts']]
activations = {i: [] for i in ['q', 'k', 'v', 'z']}
for b in tqdm(range(0, len(prompts), bs)):
    tokens = model.to_tokens(prompts[b:b+bs])
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    for key in activations.keys():
        activations[key].append(cache.stack_activation(key))

activations = {key: torch.cat(values, 1) for key, values in activations.items()}
mean_activations = {key: torch.mean(values, 1) for key, values in activations.items()} # [l pos h dm]
task_df = ioi_circuit.create_task_df()
io_vec, s_vec, pos_vec = supervised_dictionary(task_df, activations)

all_nodes_labels = ['IH+DTH-z', 'SIH-v', 'SIH-z', 'bNMH-q', 'bNMH-qk', 'bNMH-z']
all_nodes = [['IH.z', 'DTH.z'], ['SIH.v'], ['SIH.z'], ['bNMH.q'], ['bNMH.qk'], ['bNMH.z']]
attributes = ['S', 'S', 'S', 'IO', 'IO', 'IO']

for node_names, node_label, attribute in zip(all_nodes, all_nodes_labels, attributes):
    scores_df = {
        'clean_ld': [],
        'supervised_full_ld': [],
        'sae_full_ld': [],
        'supervised_average_ld': [],
        'sae_average_ld': [],
        'ablation_ld': []
    }

    for idx in tqdm(range(32)):

        example = task['prompts'][idx]

        io = example['variables']['IO']
        s = example['variables']['S2']

        pos = example['variables']['Pos']
        neg_pos = 'ABB' if pos == 'BAB' else 'BAB'

        pos_id = 0 if pos == 'ABB' else 1

        ### SUPERVISED DICTIONARY - FULL
        all_features = []
        for node in node_names:
            if attribute == 'IO':
                features = s_vec[pos][io]
            elif attribute == 'S':
                features = s_vec[pos][s]
            else:
                features = pos_vec[pos]

            all_features.append(features)

        clean_logits, patched_logits = ioi_circuit.run_with_reconstruction(
            example, 
            node_names=node_names,
            method='supervised',
            features=all_features,
            reconstruction='full',
            verbose=False
            )

        scores_df['clean_ld'].append(logits_diff(clean_logits, ' '+io, ' '+s).item())
        scores_df['supervised_full_ld'].append(logits_diff(patched_logits, ' '+io, ' '+s).item())

        ### SAE FEATURES - FULL
        clean_logits, patched_logits = ioi_circuit.run_with_reconstruction(
            example, 
            node_names=node_names,
            method='sae',
            reconstruction='full',
            verbose=False
            )
    
        scores_df['sae_full_ld'].append(logits_diff(patched_logits, ' '+io, ' '+s).item())

        ### SUPERVISED DICTIONARY - AVERAGE
        all_features = []
        for node in node_names:
            if attribute == 'IO':
                features = s_vec[pos][io]
            elif attribute == 'S':
                features = s_vec[pos][s]
            else:
                features = pos_vec[pos]

            all_features.append(features)

        clean_logits, patched_logits = ioi_circuit.run_with_reconstruction(
            example, 
            node_names=node_names,
            method='supervised',
            features=all_features,
            reconstruction='average',
            averages=mean_activations,
            verbose=False
            )

        scores_df['supervised_average_ld'].append(logits_diff(patched_logits, ' '+io, ' '+s).item())

        ### SAE FEATURES - AVERAGE
        clean_logits, patched_logits = ioi_circuit.run_with_reconstruction(
            example, 
            node_names=node_names,
            method='sae',
            reconstruction='average',
            averages=mean_activations,
            verbose=False
            )
    
        scores_df['sae_average_ld'].append(logits_diff(patched_logits, ' '+io, ' '+s).item())

        ### ABLATION
        clean_logits, patched_logits = ioi_circuit.run_with_reconstruction(
            example, 
            node_names=node_names,
            method='ablation',
            reconstruction='average',
            averages=mean_activations,
            verbose=False
            )
    
        scores_df['ablation_ld'].append(logits_diff(patched_logits, ' '+io, ' '+s).item())

    scores_df = pd.DataFrame(scores_df)
    scores_df.to_json(f'tasks/ioi/sn-scores/{attribute}_{node_label}.json')
