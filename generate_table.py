import pandas as pd
import numpy as np
from itertools import product

path_base = '/Users/claudio/Downloads/data 2'

datasets = ['yelp2018', 'amazon-book', 'gowalla']
models = ['LightGCN', 'DGCF', 'UltraGCN', 'SVDGCN']
metrics = ['ndcg', 'recall', 'precision']

couples = list(product(models, datasets))

complete = {}
for metric in metrics:
    for dataset in datasets:
        path = f'{path_base}/{dataset}/regression_{metric}_0_600.tsv'
        complete.update({f'{dataset}_{metric}': pd.read_csv(path, sep='\t')})

table = []
for model in models:
    for dataset in datasets:
        column = np.array([])
        for metric in metrics:
            data = complete.get(f'{dataset}_{metric}')
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
report.to_csv('/Users/claudio/Downloads/report_fam.tsv', sep="&", header=False, index=False)
print()
