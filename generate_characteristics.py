import os
import re
import pandas as pd
import argparse
from operator import itemgetter
from characteristics.io.loader import TsvLoader
from characteristics.io.writer import TsvWriter
from characteristics.Dataset import GraphDataset

parser = argparse.ArgumentParser(description="Run generate characteristics.")
parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--end_id', type=int, default=99)
parser.add_argument('--characteristics', type=str,
                    default='space_size_log '
                            'shape_log '
                            'density_log '
                            'gini_item gini_user '
                            'average_degree '
                            'average_degree_users '
                            'average_degree_items '
                            'average_clustering_coefficient_dot '
                            'average_clustering_coefficient_min '
                            'average_clustering_coefficient_max '
                            'average_clustering_coefficient_dot_users '
                            'average_clustering_coefficient_dot_items '
                            'average_clustering_coefficient_min_users '
                            'average_clustering_coefficient_min_items '
                            'average_clustering_coefficient_max_users '
                            'average_clustering_coefficient_max_items '
                            'average_degree_log '
                            'average_degree_users_log '
                            'average_degree_items_log '
                            'average_clustering_coefficient_dot_log '
                            'average_clustering_coefficient_min_log '
                            'average_clustering_coefficient_max_log '
                            'average_clustering_coefficient_dot_users_log '
                            'average_clustering_coefficient_dot_items_log '
                            'average_clustering_coefficient_min_users_log '
                            'average_clustering_coefficient_min_items_log '
                            'average_clustering_coefficient_max_users_log '
                            'average_clustering_coefficient_max_items_log '
                            'degree_assortativity_users '
                            'degree_assortativity_items')
parser.add_argument('--metric', type=str, default='Recall')
parser.add_argument('--splitting', type=str, default='edge-dropout node-dropout')
args = parser.parse_args()

DATA_FOLDER = os.path.abspath('./data')
OUTPUT_FOLDER = os.path.abspath('./data/')
RESULT_FOLDER = os.path.abspath('./results/')

accepted_splitting = args.splitting.split(' ')

start_id = args.start_id
end_id = args.end_id

accepted_datasets = ['gowalla', 'amazon-book', 'yelp2018']
input_dataset = args.dataset
assert input_dataset in accepted_datasets

metric = args.metric
accepted_metrics = ['Recall', 'Precision', 'nDCG']
assert metric in accepted_metrics

characteristics = args.characteristics.split(' ')

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
    performance_folder = os.path.join(RESULT_FOLDER, f'{d_info["dataset"]}_{d_info["splitting"]}_{idx}', 'performance')
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
