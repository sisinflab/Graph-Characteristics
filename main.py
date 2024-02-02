from characteristics.dataset import GraphSampler
from characteristics.io.paths import *
import csv
import random
import numpy as np

MIN_DROPOUT = 0.7
MAX_DROPOUT = 0.9

dataset_name = GOWALLA
base_seed = 42

# set the graph-based sampler
sampler = GraphSampler(dataset_name, base_seed)
sampler.set()

num_samplings = 2

print('\n\nGRAPH SAMPLING...')
with open(f'./data/{dataset_name}/sampling-stats.tsv', 'w') as f:
    fieldnames = ['dataset_id',
                  'strategy',
                  'dropout',
                  'users',
                  'items',
                  'interactions',
                  'delta_g']
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    for idx in range(num_samplings):
        sampler.set_all_seeds(base_seed + idx)

        params = {
            sampler.NODE_DROPOUT: {
                'dropout': np.random.uniform(MIN_DROPOUT, MAX_DROPOUT)
            },
            sampler.EDGE_DROPOUT: {
                'dropout': np.random.uniform(MIN_DROPOUT, MAX_DROPOUT)
            }
        }

        strategy = random.choice([sampler.NODE_DROPOUT, sampler.EDGE_DROPOUT])
        sampler.sample(strategy, params[strategy])


print()