from elliot.run import run_experiment
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

stats = pd.read_csv(f'./data/{args.dataset}/sampling-stats.tsv', sep='\t')

for idx, row in stats.iterrows():
    strategy = '-'.join(row['strategy'].split(' '))
    dataset_id = row['dataset_id']
    print(f"\n\nSTARTING TRAINING ON DATASET WITH STRATEGY: {strategy} AND ID: {dataset_id}...")
    run_experiment(f"config_files/{args.dataset}.yml", sampling=strategy, idx=dataset_id, gpu=args.gpu)
    print(f"\n\nTRAINING ENDED")
