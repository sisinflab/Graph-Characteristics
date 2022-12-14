import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description="Run collect results.")
parser.add_argument('--dataset', type=str, default='yelp2018')
parser.add_argument('--filename', type=str, default='sampling-stats-origin.tsv')
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--end_id', type=int, default=99)
parser.add_argument('--characteristics', type=str, default='users items interactions delta_g')
args = parser.parse_args()

sampling_stats = pd.read_csv(f'data/{args.dataset}/{args.filename}', sep='\t')
characteristics = args.characteristics.split(' ')
start_id = args.start_id
end_id = args.end_id
sampling_stats = sampling_stats[start_id:end_id + 1]
sampling_stats['dataset'] = args.dataset + '_' + sampling_stats['strategy'].map(
    lambda x: '-'.join(x.split(' '))) + '_' + sampling_stats['dataset_id'].map(str)
sampling_stats = sampling_stats[['dataset'] + characteristics]
sampling_stats['LightGCN'] = pd.Series([0.0] * len(sampling_stats))
sampling_stats['DGCF'] = pd.Series([0.0] * len(sampling_stats))
sampling_stats['UltraGCN'] = pd.Series([0.0] * len(sampling_stats))
sampling_stats['SVDGCN'] = pd.Series([0.0] * len(sampling_stats))

dict_metrics = dict()
dict_metrics['Recall'] = sampling_stats.copy()
dict_metrics['nDCG'] = sampling_stats.copy()
dict_metrics['Precision'] = sampling_stats.copy()

for key, value in dict_metrics.items():
    for idx, row in value.iterrows():
        all_files = os.listdir(f'results/{row["dataset"]}/performance/')
        file = [f for f in all_files if f.startswith('rec_cutoff')][0]
        current_metric_values = pd.read_csv(f'results/{row["dataset"]}/performance/{file}', sep='\t')[key].tolist()
        dict_metrics[key].at[idx, 'LightGCN'] = current_metric_values[0]
        dict_metrics[key].at[idx, 'DGCF'] = current_metric_values[1]
        dict_metrics[key].at[idx, 'UltraGCN'] = current_metric_values[2]
        dict_metrics[key].at[idx, 'SVDGCN'] = current_metric_values[3]
    dict_metrics[key].to_csv(f'data/{args.dataset}/results-{key}-{start_id}-{end_id}.tsv', sep='\t', index=None)