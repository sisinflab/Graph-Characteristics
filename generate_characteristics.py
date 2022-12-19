import multiprocessing
import re
import tqdm
import argparse
from config import *
import pandas as pd
from multiprocessing import Pool
# multiprocessing.set_start_method('fork')
from operator import itemgetter
from characteristics.io.loader import TsvLoader
from characteristics.io.writer import TsvWriter
from characteristics.Dataset import GraphDataset

parser = argparse.ArgumentParser(description="Run generate characteristics.")
parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=600)
parser.add_argument('--characteristics', type=str, nargs='+', default=ACCEPTED_CHARACTERISTICS)
parser.add_argument('--metric', type=str, default='Recall')
parser.add_argument('--splitting', type=str, nargs='+', default=ACCEPTED_SPLITTINGS)
parser.add_argument('--proc', required=False, default=multiprocessing.cpu_count(), type=int)
parser.add_argument('-mp', action='store_true')


def find_datasets(dataset: str, selected_splittings):
    assert dataset in ACCEPTED_DATASETS

    folder = os.path.join(DATA_FOLDER, input_dataset)
    splitting_strategies = os.listdir(folder)
    splitting_strategies = [s for s in splitting_strategies if s in selected_splittings]

    datasets = {int(re.findall(r'\d+', d)[0]): {'path': os.path.join(folder, s, d),
                                                'splitting': s,
                                                'dataset': input_dataset}
                for s in splitting_strategies
                for d in os.listdir(os.path.join(folder, s))}

    return datasets, folder


def compute_characteristics_on_dataset(d_info, idx):
    print(d_info)
    print(idx)
    performance_folder = os.path.join(RESULT_FOLDER, f'{d_info["dataset"]}_{d_info["splitting"]}_{idx}',
                                      'performance')
    if os.path.exists(performance_folder):
        performance_path = os.path.join(performance_folder,
                                        [p for p in os.listdir(performance_folder) if 'rec_cutoff' in p][0])
        if os.path.exists(performance_path) is False:
            print('generate_characteristic: selected dataset rec_cutoff file is missing.\n'
                  f'Dataset id: {idx}\n'
                  f'Path: {performance_path}')
            return None

        # load dataset
        loader = TsvLoader(d_info['path'])
        dataset = GraphDataset(loader.load())

        row = {'idx': idx}
        d_characteristics = {}
        iterator = tqdm.tqdm(characteristics)
        for characteristic in iterator:
            iterator.set_description(f'Computing {characteristic} for dataset {idx}')
            d_characteristics.update({characteristic: dataset.get_metric(characteristic)})

        # load recommendation performance dataset
        loader = TsvLoader(performance_path, header=0)
        recs = loader.load()

        metric_performance = dict(zip(recs['model'].map(lambda x: x.split('_')[0]), recs[metric]))
        row.update(d_characteristics)
        row.update(metric_performance)
        return row
    else:
        print('generate_characteristic: selected dataset performance are missing.\n'
              f'Dataset id: {idx}\n'
              f'Path: {performance_folder}')
        return None


def compute_characteristics(selected_data):
    # compute characteristics
    characteristics_dataset = []
    for idx in selected_data:
        row = compute_characteristics_on_dataset(selected_data[idx], idx)
        if row is not None:
            characteristics_dataset.append(row)
    return characteristics_dataset


def compute_characteristics_mp(selected_data, n_procs):
    # compute characteristics
    mp_args = ((v, k) for k, v in selected_data.items())
    with Pool(n_procs) as pool:
        characteristics_dataset = pool.starmap_async(compute_characteristics_on_dataset, mp_args)
        characteristics_dataset = characteristics_dataset.get()
    characteristics_dataset = [x for x in characteristics_dataset if x is not None]
    return characteristics_dataset


if __name__ == '__main__':

    args = parser.parse_args()

    # find datasets
    input_dataset = args.dataset
    splittings = args.splitting
    dict_datasets, dataset_folder = find_datasets(dataset=input_dataset, selected_splittings=splittings)

    start_id = args.start
    end_id = args.end

    metric = args.metric
    assert metric in ACCEPTED_METRICS

    mp = args.mp
    n_processes = args.proc

    characteristics = args.characteristics

    # select datasets from the selected range
    datasets_idx = [idx for idx in range(start_id, end_id) if idx in dict_datasets]
    assert len(datasets_idx) > 0, f'Not a single dataset found in range {start_id} {end_id}'
    if len(datasets_idx) == 1:
        selected_datasets_info = {datasets_idx[0]: dict_datasets[datasets_idx[0]]}
    else:
        selected_datasets_info = dict(zip(datasets_idx, (itemgetter(*datasets_idx)(dict_datasets))))

    # compute characteristics
    if mp:
        characteristics = compute_characteristics_mp(selected_datasets_info, n_procs=n_processes)
    else:
        characteristics = compute_characteristics(selected_datasets_info)

    # store results
    characteristics = pd.DataFrame(characteristics)
    writer = TsvWriter(main_directory=OUTPUT_FOLDER, drop_header=False)
    writer.write(characteristics,
                 file_name=f'characteristics_{metric.lower()}_{start_id}_{end_id}',
                 directory=input_dataset)
