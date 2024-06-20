import json

with open('tasks/ioi/cfg.json') as f:
    cfg = json.load(f)

names = cfg['variables'][0]['values']

def supervised_dictionary(df, activations):
    
    io_vec = {
        'ABB': {},
        'BAB': {}
    }
    s_vec = {
        'ABB': {},
        'BAB': {}
    }

    for name in names:
        io_vec['ABB'][name] = {}
        io_vec['BAB'][name] = {}
        s_vec['ABB'][name] = {}
        s_vec['BAB'][name] = {}

        for c in ['q', 'k', 'v', 'z']:
            centered_activations = activations[c] - activations[c].mean(2).mean(1)[:, None, None]
            mask = (df['IO'] == name) & (df['Pos'] == 0)
            io_vec['ABB'][name][c] = centered_activations[:, mask].mean(1)

            mask = (df['IO'] == name) & (df['Pos'] == 1)
            io_vec['BAB'][name][c] = centered_activations[:, mask].mean(1)

            mask = (df['S1'] == name) & (df['Pos'] == 0)
            s_vec['ABB'][name][c] = centered_activations[:, mask].mean(1)

            mask = (df['S1'] == name) & (df['Pos'] == 1)
            s_vec['BAB'][name][c] = centered_activations[:, mask].mean(1)

    # Pos
    pos_vec = {
        'ABB': {},
        'BAB': {}
    }

    for c in ['q', 'k', 'v', 'z']:
        centered_activations = activations[c] - activations[c].mean(2).mean(1)[:, None, None]
        mask = df['Pos'] == 0
        pos_vec['ABB'][c] = centered_activations[:, mask].mean(1)

        mask = df['Pos'] == 1
        pos_vec['BAB'][c] = centered_activations[:, mask].mean(1)
    
    return (io_vec, s_vec, pos_vec)

def sae_dictionary():
    pass