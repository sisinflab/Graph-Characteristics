import os
import pandas as pd
import argparse
import numpy as np
import random
import networkx
import torch
from networkx.algorithms import bipartite
import csv
from torch_geometric.utils.dropout import dropout_node, dropout_edge


def parse_args():
    parser = argparse.ArgumentParser(description="Run graph sampling (Node Dropout, Edge Dropout).")
    parser.add_argument('--dataset', nargs='?', default='gowalla', help='dataset name')
    parser.add_argument('--filename', nargs='?', default='dataset.tsv', help='filename')
    parser.add_argument('--sampling_strategies', nargs='+', type=str, default=['ND', 'ED'],
                        help='graph sampling strategy')
    parser.add_argument('--num_samplings', nargs='?', type=int, default=600,
                        help='number of samplings')
    parser.add_argument('--random_seed', nargs='?', type=int, default=42,
                        help='random seed for reproducibility')

    return parser.parse_args()


args = parse_args()


def calculate_statistics(data, info):
    # mapping users and items
    num_users = torch.unique(data[0]).shape[0]
    current_public_to_private_users = {u.item(): idx for idx, u in enumerate(torch.unique(data[0]))}
    current_public_to_private_items = {i.item(): idx + num_users for idx, i in enumerate(torch.unique(data[1]))}
    current_private_to_public_users = {idx: u for u, idx in current_public_to_private_users.items()}
    current_private_to_public_items = {idx: i for i, idx in current_public_to_private_items.items()}

    # rescale nodes indices to feed the edge_index into networkx
    graph = networkx.Graph()
    graph.add_nodes_from([idx for idx, _ in enumerate(torch.unique(data[0]))], bipartite='users')
    graph.add_nodes_from([idx + num_users for idx, _ in enumerate(torch.unique(data[1]))], bipartite='items')
    graph.add_edges_from(list(zip(
        [current_public_to_private_users[u] for u in data[0].tolist()],
        [current_public_to_private_items[i] for i in data[1].tolist()]))
    )

    if networkx.is_connected(graph):
        # basic statistics
        user_nodes, item_nodes = bipartite.sets(graph)
        num_users = len(user_nodes)
        num_items = len(item_nodes)
        m = len(graph.edges())
        delta_g = m / (num_users * num_items)

        stats_dict = {
            'users': num_users,
            'items': num_items,
            'interactions': m,
            'delta_g': delta_g
        }

        info.update(stats_dict)

        return info, None
    else:
        # take the subgraph with maximum extension
        graph = graph.subgraph(max(networkx.connected_components(graph), key=len))

        # basic statistics
        user_nodes, item_nodes = bipartite.sets(graph)
        num_users = len(user_nodes)
        num_items = len(item_nodes)
        m = len(graph.edges())
        delta_g = m / (num_users * num_items)

        connected_edges = list(graph.edges())
        connected_edges = [[current_private_to_public_users[i] for i, j in connected_edges],
                           [current_private_to_public_items[j] for i, j in connected_edges]]

        stats_dict = {
            'users': num_users,
            'items': num_items,
            'interactions': m,
            'delta_g': delta_g
        }

        info.update(stats_dict)

        edge_index = torch.tensor([connected_edges[0], connected_edges[1]], dtype=torch.int64)

        return info, edge_index


def set_all_seeds(current_seed):
    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    torch.cuda.manual_seed(current_seed)
    torch.cuda.manual_seed_all(current_seed)
    torch.backends.cudnn.deterministic = True


def graph_sampling():
    # load public dataset
    dataset = pd.read_csv(f'./data/{args.dataset}/{args.filename}', sep='\t', header=None)
    initial_num_users = dataset[0].nunique()
    initial_num_items = dataset[1].nunique()
    initial_users = dataset[0].unique().tolist()
    initial_items = dataset[1].unique().tolist()

    # public --> private reindexing
    public_to_private_users = {u: idx for idx, u in enumerate(initial_users)}
    public_to_private_items = {i: idx + initial_num_users for idx, i in enumerate(initial_items)}
    del initial_users, initial_items

    # private --> public reindexing
    private_to_public_users = {idx: u for u, idx in public_to_private_users.items()}
    private_to_public_items = {idx: i for i, idx in public_to_private_items.items()}

    # build undirected and bipartite graph with networkx
    graph = networkx.Graph()
    graph.add_nodes_from(list(range(initial_num_users)), bipartite='users')
    graph.add_nodes_from(list(range(initial_num_users, initial_num_users + initial_num_items)),
                         bipartite='items')
    graph.add_edges_from(list(zip(
        [public_to_private_users[u] for u in dataset[0].tolist()],
        [public_to_private_items[i] for i in dataset[1].tolist()]))
    )

    connected_graph = True

    # if graph is not connected, retain only the biggest connected portion
    if not networkx.is_connected(graph):
        graph = graph.subgraph(max(networkx.connected_components(graph), key=len))
        connected_graph = False

    # calculate statistics
    user_nodes, item_nodes = bipartite.sets(graph)
    num_users = len(user_nodes)
    num_items = len(item_nodes)
    m = len(graph.edges())
    delta_g = m / (num_users * num_items)

    if connected_graph:
        edges = [[public_to_private_users[r] for r in dataset[0].tolist()],
                 [public_to_private_items[c] for c in dataset[1].tolist()]]
        edge_index = torch.tensor(edges, dtype=torch.int64)
        del edges

    else:
        # the reindexing needs to be performed again
        connected_users = [private_to_public_users[u] for u in user_nodes]
        connected_items = [private_to_public_items[i] for i in item_nodes]
        connected_edges = list(graph.edges())
        connected_edges = [[private_to_public_users[i] for i, j in connected_edges],
                           [private_to_public_items[j] for i, j in connected_edges]]
        dataset = pd.concat([pd.Series(connected_edges[0]), pd.Series(connected_edges[1])], axis=1)
        del connected_edges

        # the public --> private reindexing is performed again
        public_to_private_users = {u: idx for idx, u in enumerate(connected_users)}
        public_to_private_items = {i: idx + num_users for idx, i in enumerate(connected_items)}
        del connected_users, connected_items

        # the private --> public reindexing is performed again
        private_to_public_users = {idx: u for u, idx in public_to_private_users.items()}
        private_to_public_items = {idx: i for i, idx in public_to_private_items.items()}

        edges = [[public_to_private_users[r] for r in dataset[0].tolist()],
                 [public_to_private_items[c] for c in dataset[1].tolist()]]
        edge_index = torch.tensor(edges, dtype=torch.int64)
        del edges

    del graph

    # print statistics
    print(f'DATASET: {args.dataset}')
    print(f'Number of users: {num_users}')
    print(f'Number of items: {num_items}')
    print(f'Number of interactions: {m}')
    print(f'Density: {delta_g}')

    filename_no_extension = args.filename.split('.')[0]
    extension = args.filename.split('.')[1]

    print('\n\nSTART GRAPH SAMPLING...')
    with open(f'./data/{args.dataset}/sampling-stats.tsv', 'w') as f:
        fieldnames = ['dataset_id',
                      'strategy',
                      'dropout',
                      'users',
                      'items',
                      'interactions',
                      'delta_g']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for idx in range(args.num_samplings):
            set_all_seeds(args.random_seed + idx)
            gss = random.choice(args.sampling_strategies)
            dr = np.random.uniform(0.7, 0.9)
            if gss == 'ND':
                if not os.path.exists(f'./data/{args.dataset}/node-dropout/'):
                    os.makedirs(f'./data/{args.dataset}/node-dropout/')
                print(f'\n\nRunning NODE DROPOUT with dropout ratio {dr}')
                sampled_edge_index, _, _ = dropout_node(edge_index, p=dr, num_nodes=num_users + num_items)
                current_stats_dict, sampled_graph = calculate_statistics(sampled_edge_index,
                                                                         info={'dataset_id': idx,
                                                                               'strategy': 'node dropout',
                                                                               'dropout': dr})
                if sampled_graph is not None:
                    sampled_rows = [private_to_public_users[r] for r in sampled_graph[0].tolist()]
                    sampled_cols = [private_to_public_items[c] for c in sampled_graph[1].tolist()]
                else:
                    sampled_rows = [private_to_public_users[r] for r in sampled_edge_index[0].tolist()]
                    sampled_cols = [private_to_public_items[c] for c in sampled_edge_index[1].tolist()]
                sampled_dataset = pd.concat([pd.Series(sampled_rows), pd.Series(sampled_cols)], axis=1)
                sampled_dataset.to_csv(
                    f'./data/{args.dataset}/node-dropout/{filename_no_extension}-{idx}.{extension}',
                    sep='\t', header=None, index=None)
                writer.writerow(current_stats_dict)
            elif gss == 'ED':
                if not os.path.exists(f'./data/{args.dataset}/edge-dropout/'):
                    os.makedirs(f'./data/{args.dataset}/edge-dropout/')
                print(f'\n\nRunning EDGE DROPOUT with dropout ratio {dr}')
                sampled_edge_index, _ = dropout_edge(edge_index, p=dr)
                current_stats_dict, sampled_graph = calculate_statistics(sampled_edge_index,
                                                                         info={'dataset_id': idx,
                                                                               'strategy': 'edge dropout',
                                                                               'dropout': dr})
                if sampled_graph is not None:
                    sampled_rows = [private_to_public_users[r] for r in sampled_graph[0].tolist()]
                    sampled_cols = [private_to_public_items[c] for c in sampled_graph[1].tolist()]
                else:
                    sampled_rows = [private_to_public_users[r] for r in sampled_edge_index[0].tolist()]
                    sampled_cols = [private_to_public_items[c] for c in sampled_edge_index[1].tolist()]
                sampled_dataset = pd.concat([pd.Series(sampled_rows), pd.Series(sampled_cols)], axis=1)
                sampled_dataset.to_csv(
                    f'./data/{args.dataset}/edge-dropout/{filename_no_extension}-{idx}.{extension}',
                    sep='\t', header=None, index=None)
                writer.writerow(current_stats_dict)
            else:
                raise NotImplementedError('This graph sampling strategy has not been implemented yet!')
    print('\n\nEND GRAPH SAMPLING...')


if __name__ == '__main__':
    graph_sampling()
