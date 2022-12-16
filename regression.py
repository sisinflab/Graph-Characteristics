import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser(description="Run regression.")
parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--end_id', type=int, default=99)
parser.add_argument('--characteristics', type=str, default='space_size_log shape_log density_log gini_item gini_user')
parser.add_argument('--metric', type=str, default='recall')
args = parser.parse_args()

results = pd.read_csv(f'data/{args.dataset}/characteristics_{args.metric}_{args.start_id}_{args.end_id}.tsv', sep='\t')
models = ['LightGCN', 'DGCF', 'UltraGCN', 'SVDGCN']
characteristics = args.characteristics.split(' ')
characteristics_w = ['intercept'] + [c + '_w' for c in characteristics]

models_results = []

for idx, m in enumerate(models):
    X = results[characteristics].to_numpy()
    y = results[m].to_numpy()
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)
    adjusted_score = 1 - (1 - score) * ((X.shape[0] - 1) / (X.shape[0] - len(characteristics) - 1))
    coefs = [reg.intercept_] + reg.coef_.tolist()
    models_results.append({
        'model': m,
        'score': score,
        'adjusted_score': adjusted_score
    })

    for idxx, c in enumerate(characteristics):
        models_results[idx][c] = X.reshape(-1)[idxx]
    for idxx, c in enumerate(characteristics_w):
        models_results[idx][c] = coefs[idxx]

df = pd.DataFrame.from_dict(models_results)
df.to_csv(f'data/{args.dataset}/regression-{args.metric}-{args.start_id}-{args.end_id}.tsv', sep='\t', index=None)
