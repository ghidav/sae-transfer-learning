import pandas as pd

data = pd.read_json('training/alpaca_cleaned.json')
data = data[data['input'] == ''].sample(10300, random_state=42)

template = """User: {x}\nAssistant: {y}"""

def apply_template(x):
    return template.format(
        x=x['instruction'],
        y=x['output']
    )

data['prompt'] = data.apply(apply_template, axis=1)

train = data.iloc[:10000]
eval = data.iloc[10000:]

train.to_csv('training/train.csv')
eval.to_csv('training/eval.csv')