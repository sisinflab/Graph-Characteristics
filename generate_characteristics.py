import os
import re
import pandas as pd
from operator import itemgetter
from characteristics.io.loader import TsvLoader
from characteristics.io.writer import TsvWriter
from characteristics.Dataset import GraphDataset

DATA_FOLDER = os.path.abspath('./data')
OUTPUT_FOLDER = os.path.abspath('./data/')
RESULT_FOLDER = os.path.abspath('./results/')

accepted_splitting = ['edge-dropout', 'node-dropout']

# prendere in input un range di id di dataset
start_id = 599
end_id = 610

# prendere in input il nome del dataset
accepted_datasets = ['gowalla', 'amazon-book', 'yelp2018']
input_dataset = 'gowalla'
assert input_dataset in accepted_datasets

metric = 'Recall'
accepted_metrics = ['Recall', 'Precision', 'nDCG']
assert metric in accepted_metrics

# lista delle caratteristiche
characteristics = ['transactions', 'space_size', 'space_size_log', 'shape', 'shape_log', 'density', 'density_log',
                   'gini_item', 'gini_user']

# cerca i dataset generati
dataset_folder = os.path.join(DATA_FOLDER, input_dataset)

splitting_strategies = os.listdir(dataset_folder)
splitting_strategies = [s for s in splitting_strategies if s in accepted_splitting]

dict_datasets = {int(re.findall(r'\d+', d)[0]): {'path': os.path.join(dataset_folder, s, d),
                                                 'splitting': s,
                                                 'dataset': input_dataset}
                 for s in splitting_strategies
                 for d in os.listdir(os.path.join(dataset_folder, s))}

# get elements
datasets_idx = [idx for idx in range(start_id, end_id) if idx in dict_datasets]
assert len(datasets_idx) > 0, f'Not a single dataset found in range {start_id} {end_id}'
if len(datasets_idx) == 1:
    selected_datasets_info = {datasets_idx[0]: dict_datasets[datasets_idx[0]]}
else:
    selected_datasets_info = dict(zip(datasets_idx, (itemgetter(*datasets_idx)(dict_datasets))))

characteristics_dataset = []
for idx, d_info in selected_datasets_info.items():
    performance_folder = os.path.join(RESULT_FOLDER, f'{d_info["dataset"]}_'
                                                   f'{d_info["splitting"]}_'
                                                   f'{idx}', 'performance')
    if os.path.exists(performance_folder):
        performance_path = os.path.join(performance_folder,
                                        [p for p in os.listdir(performance_folder) if 'rec_cutoff' in p][0])
        if os.path.exists(performance_path) is False:
            print('generate_characteristic: selected dataset rec_cutoff file is missing.\n'
                  f'Dataset id: {idx}\n'
                  f'Path: {performance_path}')
            continue

        loader = TsvLoader(d_info['path'])
        dataset = GraphDataset(loader.load())
        row = {'idx': idx}
        d_characteristics = {c: dataset.get_metric(c) for c in characteristics}

        loader = TsvLoader(performance_path, header=0)
        recs = loader.load()
        metric_performance = dict(zip(recs['model'].map(lambda x: x.split('_')[0]), recs[metric]))
        row.update(d_characteristics)
        row.update(metric_performance)
        characteristics_dataset.append(row)
    else:
        print('generate_characteristic: selected dataset performance are missing.\n'
              f'Dataset id: {idx}\n'
              f'Path: {performance_folder}')
        continue

characteristics_dataset = pd.DataFrame(characteristics_dataset)
writer = TsvWriter(main_directory=OUTPUT_FOLDER, drop_header=False)
writer.write(characteristics_dataset,
             file_name=f'characteristics_{metric.lower()}_{start_id}_{end_id}',
             directory=input_dataset)
