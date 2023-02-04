import pandas as pd
import numpy as np
from itertools import product

datasets = ['gowalla']
models = ['LightGCN', 'DGCF', 'UltraGCN', 'SVDGCN']
metrics = ['recall']
samplings = []

alpha = 0.0

couples = list(product(models, datasets, metrics))

complete = {}
for metric in metrics:
    for dataset in datasets:
        path = f'./data/{dataset}/regression_{alpha}_{metric}_0_600.tsv'
        complete.update({f'{metric}': pd.read_csv(path, sep='\t')})

table = []
for metric in metrics:
    for model in models:
        column = np.array([])
        data = complete.get(f'{metric}')
        data_model = data[data['model'] == model]
        data_scores = [f"{item:.3f}" for item in data_model.iloc[:, 1:3].values.squeeze()]
        data_chars = [f"{item:.3f}" for item in data_model.iloc[:, 3:15].values.squeeze()]
        p_values = data_model.iloc[:, 15:].values.squeeze()
        p_values = list(map(lambda p: '***' if p <= 0.001 else '**' if p <= 0.01 else '*' if p <= 0.05 else '',
                            p_values))
        data_p_values = list(zip(data_chars, p_values))
        data_p_values_formatted = [f'${v}^{{{p}}}$' for v, p in data_p_values]
        data_scores_formatted = [f'${v1} ({v2})$' for v1, v2 in [tuple(data_scores)]]
        column = np.r_[column, data_scores_formatted + data_p_values_formatted]
        table.append(column.tolist())

report = pd.DataFrame(table).T
report.to_csv(f'./data/{datasets[0]}/table_{alpha}_rq2.tsv', sep="&", header=False, index=False)
print()
