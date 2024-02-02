import math
import pandas as pd
import numpy as np
from characteristics.io.paths import *
import random
import torch
from torch_geometric.utils.dropout import dropout_node, dropout_edge

# graph libraries
import networkx
from networkx.algorithms import bipartite


class RecommendationDataset:

    def __init__(self, data: pd.DataFrame, user_col=None, item_col=None, rating_col=None, timestamp_col=None):

        assert isinstance(data, pd.DataFrame)
        self._dataset = data
        self._is_private = False

        n_columns = len(data.columns)
        assert n_columns >= 2, f'{self.__class__.__name__}: the columns must be at least two.'

        # user and item column
        if user_col is None:
            user_col = 0
        if item_col is None:
            item_col = 1

        # if ratings column is missing, it is assumed that ratings are implicit
        self._is_implicit = False
        if (len(data.columns)) < 3 and (rating_col is None):
            rating_col = 'ratings'
            data[rating_col] = [1] * len(data)
            self._is_implicit = True

        self._names = data.columns.to_list()
        self.user_col = None
        self.item_col = None
        self.rating_col = None
        self.timestamp_col = None
        self.set_columns(user_col, item_col, rating_col, timestamp_col)

        # -- users and items setting --
        self._sorted_users = None
        self._sorted_items = None

        # map users and items with a 0-indexed mapping
        self._public_to_private_users = self.public_to_private(self._dataset[self.user_col].unique().tolist())
        self._n_users = len(self._public_to_private_users)

        self._public_to_private_items = self.public_to_private(self._dataset[self.item_col].unique().tolist(),
                                                               self._n_users)
        self._n_items = len(self._public_to_private_items)

        self._private_to_public_users = self.private_to_public(self._public_to_private_users)
        self._private_to_public_items = self.private_to_public(self._public_to_private_items)

        self.to_private()

        self._users = list(range(self._n_users))
        self._items = list(range(self._n_items))

        # -- ratings setting --

        # metrics
        self._transactions = None
        self._space_size = None
        self._space_size_log = None
        self._shape = None
        self._shape_log = None
        self._density = None
        self._density_log = None
        self._gini_item = None
        self._gini_user = None
        self.metrics = ['transactions', 'space_size', 'space_size_log', 'shape', 'shape_log', 'density', 'density_log',
                        'gini_item', 'gini_user',
                        'average_degree', 'average_degree_users', 'average_degree_items',
                        'average_degree_log', 'average_degree_users_log', 'average_degree_items_log',
                        'average_clustering_coefficient_dot',
                        'average_clustering_coefficient_min',
                        'average_clustering_coefficient_max',
                        'average_clustering_coefficient_dot_log',
                        'average_clustering_coefficient_min_log',
                        'average_clustering_coefficient_max_log',
                        'average_clustering_coefficient_dot_users', 'average_clustering_coefficient_dot_items',
                        'average_clustering_coefficient_min_users', 'average_clustering_coefficient_min_items',
                        'average_clustering_coefficient_max_users', 'average_clustering_coefficient_max_items',
                        'average_clustering_coefficient_dot_users_log', 'average_clustering_coefficient_dot_items_log',
                        'average_clustering_coefficient_min_users_log', 'average_clustering_coefficient_min_items_log',
                        'average_clustering_coefficient_max_users_log', 'average_clustering_coefficient_max_items_log',
                        'degree_assortativity_users', 'degree_assortativity_items']
        self._ratings_per_user = None
        self._ratings_per_item = None

    @property
    def n_users(self):
        self._n_users = int(self._dataset[self.user_col].nunique())
        return self._n_users

    @property
    def n_items(self):
        self._n_items = int(self._dataset[self.item_col].nunique())
        return self._n_items

    def set_columns(self, user_col, item_col, rating_col, timestamp_col):

        new_col_names = {col: name for col, name in zip([user_col, item_col, rating_col, timestamp_col],
                                                        ['user', 'item', 'rating', 'timestamp'])}
        self._dataset.rename(columns=new_col_names, inplace=True)

        for var, col in zip(['user_col', 'item_col', 'rating_col', 'timestamp_col'], self._dataset.columns):
            self.__setattr__(var, col)

    @property
    def sorted_items(self):
        if self._sorted_items is None:
            count_items = self._dataset.groupby(self.item_col).count().sort_values(by=[self.user_col])
            self._sorted_items = dict(zip(count_items.index, count_items[self.user_col]))
        return self._sorted_items

    @property
    def sorted_users(self):
        if self._sorted_users is None:
            count_users = self._dataset.groupby(self.user_col).count().sort_values(by=[self.item_col])
            self._sorted_users = dict(zip(count_users.index, count_users[self.item_col]))
        return self._sorted_users

    # --- MAPPING FUNCTIONS ---
    @property
    def transactions(self):
        if self._transactions is None:
            self._transactions = len(self._dataset)
        return self._transactions

    @staticmethod
    def public_to_private(lst, offset=0):
        return {el: idx + offset for idx, el in enumerate(lst)}

    @staticmethod
    def private_to_public(pub_to_prvt: dict):
        mapping = {el: idx for idx, el in pub_to_prvt.items()}
        if len(pub_to_prvt) != len(mapping):
            print('WARNING: private to public mapping could be incorrect. Please, check your code.')
        return mapping

    def map_dataset(self, user_mapping, item_mapping):
        self._dataset[self.user_col] = self._dataset[self.user_col].map(user_mapping)
        self._dataset[self.item_col] = self._dataset[self.item_col].map(item_mapping)

    def to_public(self):
        if self._is_private:
            self.map_dataset(self._private_to_public_users, self._private_to_public_items)
        self._is_private = False

    def to_private(self):
        if not self._is_private:
            self.map_dataset(self._public_to_private_users, self._public_to_private_items)
        self._is_private = True

    def to_csv(self, path, **kwargs):
        self._dataset.to_csv(path, **kwargs)

    # -- METRICS --

    def get_metric(self, metric):
        assert metric in self.metrics, f'{self.__class__.__name__}: metric \'{metric}\' not found.'
        func = getattr(self, metric)
        return func()

    def space_size(self):
        if self._space_size is None:
            scale_factor = 1000
            self._space_size = math.sqrt(self._n_users * self._n_items) / scale_factor
        return self._space_size

    def space_size_log(self):
        if self._space_size_log is None:
            self._space_size_log = math.log10(self.space_size())
        return self._space_size_log

    def shape(self):
        if self._shape is None:
            self._shape = self._n_users / self._n_items
        return self._shape

    def shape_log(self):
        if self._shape_log is None:
            self._shape_log = math.log10(self.shape())
        return self._shape_log

    def density(self):
        if self._density is None:
            self._density = self.transactions / (self._n_users * self._n_items)
        return self._density

    def density_log(self):
        if self._density_log is None:
            self._density_log = math.log10(self.density())
        return self._density_log

    @staticmethod
    def gini(x):
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x) ** 2 * np.mean(x))

    def gini_item(self):
        if self._gini_item is None:
            self._gini_item = self.gini(np.array(list(self.sorted_items.values())))
        return self._gini_item

    def gini_user(self):
        if self._gini_user is None:
            self._gini_user = self.gini(np.array(list(self.sorted_users.values())))
        return self._gini_user

    def ratings_per_user(self):

        if self._ratings_per_user is None:
            self._ratings_per_user = self.transactions / self.n_users
        return self._ratings_per_user

    def ratings_per_item(self):

        if self._ratings_per_item is None:
            self._ratings_per_item = self.transactions / self.n_items
        return self._ratings_per_item


class GraphDataset(RecommendationDataset):

    def __init__(self, data: pd.DataFrame):
        assert isinstance(data, pd.DataFrame)

        super(GraphDataset, self).__init__(data)

        self.graph = self.networkx_bipartite_graph()
        self.num_edges = len(self.graph.edges)
        self.user_nodes, self.item_nodes = bipartite.sets(self.bipartite_graph)
        self.num_user_nodes, self.num_item_nodes = len(self.user_nodes), len(self.item_nodes)
        self.user_graph, self.item_graph = None, None

        # metrics

        # degree
        self._average_degree = None
        self._average_degree_users = None
        self._average_degree_items = None
        self._average_degree_log = None
        self._average_degree_users_log = None
        self._average_degree_items_log = None

        # clustering
        self._average_clustering_coefficient_dot = None
        self._average_clustering_coefficient_min = None
        self._average_clustering_coefficient_max = None
        self._average_clustering_coefficient_dot_users = None
        self._average_clustering_coefficient_dot_items = None
        self._average_clustering_coefficient_min_users = None
        self._average_clustering_coefficient_min_items = None
        self._average_clustering_coefficient_max_users = None
        self._average_clustering_coefficient_max_items = None
        self._average_clustering_coefficient_dot_log = None
        self._average_clustering_coefficient_min_log = None
        self._average_clustering_coefficient_max_log = None
        self._average_clustering_coefficient_dot_users_log = None
        self._average_clustering_coefficient_dot_items_log = None
        self._average_clustering_coefficient_min_users_log = None
        self._average_clustering_coefficient_min_items_log = None
        self._average_clustering_coefficient_max_users_log = None
        self._average_clustering_coefficient_max_items_log = None

        # assortativity
        self._degree_assortativity = None
        self._degree_assortativity_users = None
        self._degree_assortativity_items = None

    def average_degree(self):

        if self._average_degree is None:
            self._average_degree = (2 * self.num_edges) / (self.num_user_nodes + self.num_item_nodes)

        return self._average_degree

    def average_degree_log(self):

        if self._average_degree_log is None:
            self._average_degree_log = math.log10(self.average_degree())

        return self._average_degree_log

    def average_degree_users(self):

        if self._average_degree_users is None:
            self._average_degree_users = self.num_edges / self.num_user_nodes

        return self._average_degree_users

    def average_degree_users_log(self):

        if self._average_degree_users_log is None:
            self._average_degree_users_log = math.log10(self.average_degree_users())

        return self._average_degree_users_log

    def average_degree_items(self):

        if self._average_degree_items is None:
            self._average_degree_items = self.num_edges / self.num_item_nodes

        return self._average_degree_items

    def average_degree_items_log(self):

        if self._average_degree_items_log is None:
            self._average_degree_items_log = math.log10(self.average_degree_items())

        return self._average_degree_items_log

    def average_clustering_coefficient_dot(self):

        if self._average_clustering_coefficient_dot is None:
            self._average_clustering_coefficient_dot = bipartite.average_clustering(self.bipartite_graph,
                                                                                    mode='dot')
        return self._average_clustering_coefficient_dot

    def average_clustering_coefficient_dot_log(self):

        if self._average_clustering_coefficient_dot_log is None:
            self._average_clustering_coefficient_dot_log = math.log10(self.average_clustering_coefficient_dot())
        return self._average_clustering_coefficient_dot_log

    def average_clustering_coefficient_min(self):

        if self._average_clustering_coefficient_min is None:
            self._average_clustering_coefficient_min = bipartite.average_clustering(self.bipartite_graph,
                                                                                    mode='min')
        return self._average_clustering_coefficient_min

    def average_clustering_coefficient_min_log(self):

        if self._average_clustering_coefficient_min_log is None:
            self._average_clustering_coefficient_min_log = math.log10(self.average_clustering_coefficient_min())
        return self._average_clustering_coefficient_min_log

    def average_clustering_coefficient_max(self):

        if self._average_clustering_coefficient_max is None:
            self._average_clustering_coefficient_max = bipartite.average_clustering(self.bipartite_graph,
                                                                                    mode='max')
        return self._average_clustering_coefficient_max

    def average_clustering_coefficient_max_log(self):

        if self._average_clustering_coefficient_max_log is None:
            self._average_clustering_coefficient_max_log = math.log10(self.average_clustering_coefficient_max())
        return self._average_clustering_coefficient_max_log

    def average_clustering_coefficient_dot_users(self):

        if self._average_clustering_coefficient_dot_users is None:
            self._average_clustering_coefficient_dot_users = bipartite.average_clustering(self.bipartite_graph,
                                                                                          mode='dot',
                                                                                          nodes=self.user_nodes)
        return self._average_clustering_coefficient_dot_users

    def average_clustering_coefficient_dot_users_log(self):

        if self._average_clustering_coefficient_dot_users_log is None:
            self._average_clustering_coefficient_dot_users_log = math.log10(
                self.average_clustering_coefficient_dot_users()
            )
        return self._average_clustering_coefficient_dot_users_log

    def average_clustering_coefficient_dot_items(self):

        if self._average_clustering_coefficient_dot_items is None:
            self._average_clustering_coefficient_dot_items = bipartite.average_clustering(self.bipartite_graph,
                                                                                          mode='dot',
                                                                                          nodes=self.item_nodes)
        return self._average_clustering_coefficient_dot_items

    def average_clustering_coefficient_dot_items_log(self):

        if self._average_clustering_coefficient_dot_items_log is None:
            self._average_clustering_coefficient_dot_items_log = math.log10(
                self.average_clustering_coefficient_dot_items()
            )
        return self._average_clustering_coefficient_dot_items_log

    def average_clustering_coefficient_min_users(self):

        if self._average_clustering_coefficient_min_users is None:
            self._average_clustering_coefficient_min_users = bipartite.average_clustering(self.bipartite_graph,
                                                                                          mode='min',
                                                                                          nodes=self.user_nodes)
        return self._average_clustering_coefficient_min_users

    def average_clustering_coefficient_min_users_log(self):

        if self._average_clustering_coefficient_min_users_log is None:
            self._average_clustering_coefficient_min_users_log = math.log10(
                self.average_clustering_coefficient_min_users()
            )
        return self._average_clustering_coefficient_min_users_log

    def average_clustering_coefficient_min_items(self):

        if self._average_clustering_coefficient_min_items is None:
            self._average_clustering_coefficient_min_items = bipartite.average_clustering(self.bipartite_graph,
                                                                                          mode='min',
                                                                                          nodes=self.item_nodes)
        return self._average_clustering_coefficient_min_items

    def average_clustering_coefficient_min_items_log(self):

        if self._average_clustering_coefficient_min_items_log is None:
            self._average_clustering_coefficient_min_items_log = math.log10(
                self.average_clustering_coefficient_min_items()
            )
        return self._average_clustering_coefficient_min_items_log

    def average_clustering_coefficient_max_users(self):

        if self._average_clustering_coefficient_max_users is None:
            self._average_clustering_coefficient_max_users = bipartite.average_clustering(self.bipartite_graph,
                                                                                          mode='max',
                                                                                          nodes=self.user_nodes)
        return self._average_clustering_coefficient_max_users

    def average_clustering_coefficient_max_users_log(self):

        if self._average_clustering_coefficient_max_users_log is None:
            self._average_clustering_coefficient_max_users_log = math.log10(
                self.average_clustering_coefficient_max_users()
            )
        return self._average_clustering_coefficient_max_users_log

    def average_clustering_coefficient_max_items(self):

        if self._average_clustering_coefficient_max_items is None:
            self._average_clustering_coefficient_max_items = bipartite.average_clustering(self.bipartite_graph,
                                                                                          mode='max',
                                                                                          nodes=self.item_nodes)
        return self._average_clustering_coefficient_max_items

    def average_clustering_coefficient_max_items_log(self):

        if self._average_clustering_coefficient_max_items_log is None:
            self._average_clustering_coefficient_max_items_log = math.log10(
                self.average_clustering_coefficient_max_items()
            )
        return self._average_clustering_coefficient_max_items_log

    def degree_assortativity_users(self):

        if self._degree_assortativity_users is None and self.user_graph is None:
            self.user_graph = bipartite.projected_graph(self.bipartite_graph, self.user_nodes)
            self._degree_assortativity_users = networkx.degree_pearson_correlation_coefficient(self.user_graph,
                                                                                               nodes=self.user_nodes)

        return self._degree_assortativity_users

    def degree_assortativity_items(self):

        if self._degree_assortativity_items is None and self.item_graph is None:
            self.item_graph = bipartite.projected_graph(self.bipartite_graph, self.item_nodes)
            self._degree_assortativity_items = networkx.degree_pearson_correlation_coefficient(self.item_graph,
                                                                                               nodes=self.item_nodes)

        return self._degree_assortativity_items

    def networkx_bipartite_graph(self):
        # build undirected and bipartite graph with networkx
        print(f'{self.__class__.__name__}: building a bipartite graph with networkx')
        graph = networkx.Graph()
        graph.add_nodes_from(self._users, bipartite='users')
        graph.add_nodes_from(self._items, bipartite='items')
        graph.add_edges_from(list(zip(self._dataset[self.user_col], self._dataset[self.item_col])))
        return graph

    def connected_graph(self):
        graph = self.networkx_bipartite_graph()
        # if graph is not connected, retain only the biggest connected portion
        if not networkx.is_connected(graph):
            print(f'{self.__class__.__name__}: the graph is not connected. Building the connected subgraph.')
            graph = graph.subgraph(max(networkx.connected_components(graph), key=len))
        print(f'{self.__class__.__name__}: graph is connected.')
        user_nodes, item_nodes = bipartite.sets(graph)
        print(f'{self.__class__.__name__}: graph characteristics'
              f'\n nodes: {len(graph.nodes)}'
              f'\n edges: {len(graph.edges)}'
              f'\n user nodes: {len(user_nodes)}'
              f'\n item nodes: {len(item_nodes)}')
        return graph

    def make_connected(self):
        self.bipartite_graph = self.connected_graph()

        user_nodes, item_nodes = bipartite.sets(self.bipartite_graph)

        n_old_users = self._n_users
        n_old_items = self._n_items

        # filter out users and items
        self._dataset = self._dataset[
            self._dataset[self.user_col].isin(user_nodes) & self._dataset[self.item_col].isin(item_nodes)]
        assert user_nodes == set(self._dataset[self.user_col].unique()), f'{self.__class__.__name__}:' \
                                                                         f' a problem occurred during dataset filtering'
        assert item_nodes == set(self._dataset[self.item_col].unique()), f'{self.__class__.__name__}:' \
                                                                         f' a problem occurred during dataset filtering'

        self._n_users = len(user_nodes)
        self._n_items = len(item_nodes)

        print(f'{self.__class__.__name__}: {n_old_users - self._n_users} users removed')
        print(f'{self.__class__.__name__}: {n_old_items - self._n_items} items removed')


class GraphSampler:

    def __init__(self, dataset_name, random_seed):

        self.__dataset_name = dataset_name
        self.__seed = random_seed

        self._data = None

        self._public_to_private_users = None
        self._public_to_private_items = None

        self._private_to_public_users = None
        self._private_to_public_items = None

        self._n_users = None
        self._n_items = None

        self._graph = None
        self._graph2 = None
        self._edge_index = None

        self.NODE_DROPOUT = 'node dropout'
        self.EDGE_DROPOUT = 'edge dropout'

        self.GRAPH_SAMPLING_STRATEGIES = [self.NODE_DROPOUT, self.EDGE_DROPOUT]
        self.__GRAPH_SAMPLING_FUNCTIONS = {
            self.NODE_DROPOUT: None,
            self.EDGE_DROPOUT: None
        }


    @property
    def data(self):
        if self._data is None:
            print('Data has not been loaded already. Please, call the method .load() before.')
            return False
        return self._data

    def is_connected(self):
        if self._graph is None:
            print('Graph has not been set. Please, call the method .set_graph() to set the graph.')
            return False
        return networkx.is_connected(self._graph)

    def load(self):
        self._data = pd.read_csv(dataset_filepath(self.__dataset_name), sep='\t', header=None)
        print(f'Dataset loaded from \'{dataset_filepath(self.__dataset_name)}\'')

    def set_users_and_items(self):
        if self.data is False:
            return None
        n_users = self._data[0].nunique()
        n_items = self._data[1].nunique()
        users = self._data[0].unique().tolist()
        items = self._data[1].unique().tolist()

        print(f'Found {n_users} users')
        print(f'Found {n_items} items')
        self._n_users = n_users
        self._n_items = n_items

        return n_users, n_items, users, items

    def user_and_item_indexing(self):
        if self.data is False:
            return None

        n_users, n_items, users, items = self.set_users_and_items()

        # public --> private reindexing
        self._public_to_private_users = {u: idx for u, idx in zip(users, range(n_users))}
        self._public_to_private_items = {i: idx for i, idx in zip(items, range(n_users, n_users + n_items))}
        del users, items

        # private --> public reindexing
        self._private_to_public_users = {idx: u for u, idx in self._public_to_private_users.items()}
        self._private_to_public_items = {idx: i for i, idx in self._public_to_private_items.items()}
        return None

    def set(self):
        # load data if have not already been loaded
        if self.data is False:
            self.load()
        # derive users and data from the dataset
        self.user_and_item_indexing()

        self.set_graph()

        # check if the graph is connected. If not, reconstruct the graph
        if self.retain_connected_edges():
            self.user_and_item_indexing()
            self.set_graph()

    def set_graph(self):
        if self._public_to_private_users is None or self._public_to_private_items is None:
            self.user_and_item_indexing()

        self._graph2 = GraphDataset(self._data)

        # build undirected and bipartite graph with networkx
        self._graph = networkx.Graph()
        self._graph.add_nodes_from(list(self._public_to_private_users.values()),
                                   bipartite='users')
        self._graph.add_nodes_from(list(self._public_to_private_items.values()),
                                   bipartite='items')
        self._graph.add_edges_from(list(zip(
            [self._public_to_private_users[u] for u in self._data[0].tolist()],
            [self._public_to_private_items[i] for i in self._data[1].tolist()])))

        users, items = list(zip(*(self._graph.edges())))
        self._edge_index = torch.tensor([users, items], dtype=torch.int64)
        return None

    def retain_connected_edges(self):
        if self._graph is None:
            # create the graph if not built before
            self.set_graph()

        # if graph is not connected, retain only the biggest connected portion
        if not self.is_connected():
            print('Graph not connected. Selecting the largest connected subgraph.')
            self._graph = self._graph.subgraph(max(networkx.connected_components(self._graph), key=len))
            connected_users, connected_items = bipartite.sets(self._graph)
            connected_edges = list(self._graph.edges())

            print(f'N. users: {len(connected_users)}')
            print(f'N. items: {len(connected_items)}')
            print(f'N. edges: {len(connected_edges)}')

            public_connected_edges = list(
                map(lambda x: (self._private_to_public_users[x[0]], self._private_to_public_items[x[1]]),
                    connected_edges))

            dataset = pd.concat([pd.Series(public_connected_edges[0]), pd.Series(public_connected_edges[1])], axis=1)
            del connected_edges
            self._data = dataset
            return True

        return False

    def set_all_seeds(self, seed=None):
        if seed is None:
            self.__seed = seed
        random.seed(self.__seed)
        np.random.seed(self.__seed)
        torch.manual_seed(self.__seed)
        torch.cuda.manual_seed(self.__seed)
        torch.cuda.manual_seed_all(self.__seed)
        torch.backends.cudnn.deterministic = True


    def sample(self, strategy, params: dict):
        assert strategy in self.GRAPH_SAMPLING_STRATEGIES

    def random_sample(self, strategies: list):
        # pick randomly a strategy
        strategy = random.choice(strategies)

    def node_dropout(self, dropout):
        if self._edge_index is None:
            raise AttributeError('_edge_index not found.\n'
                                 'The graph has not been set. Please, see .set() or .set_graph() methods')

        print(f'\n\nRunning NODE DROPOUT with dropout ratio {dropout}')
        sampled_edge_index, _, _ = dropout_node(self._edge_index, p=dropout, num_nodes=self._n_users + self._n_items)
        # current_stats_dict, sampled_graph = calculate_statistics(sampled_edge_index,
        #                                                          info={'dataset_id': idx,
        #                                                                'strategy': 'node dropout',
        #                                                                'dropout': dr})
        # if sampled_graph is not None:
        #     sampled_rrandom_sampleows = [private_to_public_users[r] for r in sampled_graph[0].tolist()]
        #     sampled_cols = [private_to_public_items[c] for c in sampled_graph[1].tolist()]
        # else:
        #     sampled_rows = [private_to_public_users[r] for r in sampled_edge_index[0].tolist()]
        #     sampled_cols = [private_to_public_items[c] for c in sampled_edge_index[1].tolist()]
        # sampled_dataset = pd.concat([pd.Series(sampled_rows), pd.Series(sampled_cols)], axis=1)
        # sampled_dataset.to_csv(f'./data/{dataset_name}/node-dropout/{filename_no_extension}-{idx}.{extension}',
        #                        sep='\t', header=False, index=False)
        # writer.writerow(current_stats_dict)


