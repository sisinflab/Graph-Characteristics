import os

DATA_FOLDER = os.path.abspath('./data')
OUTPUT_FOLDER = os.path.abspath('./data/')
RESULT_FOLDER = os.path.abspath('./results/')
ACCEPTED_DATASETS = ['gowalla', 'amazon-book', 'yelp2018']
ACCEPTED_SPLITTINGS = ['edge-dropout', 'node-dropout']
ACCEPTED_METRICS = ['Recall', 'Precision', 'nDCG']
ACCEPTED_CHARACTERISTICS = ['space_size_log',
                            'shape_log',
                            'density_log',
                            'gini_item',
                            'gini_user',
                            'average_degree',
                            'average_degree_users',
                            'average_degree_items',
                            'average_clustering_coefficient_dot',
                            'average_clustering_coefficient_min',
                            'average_clustering_coefficient_max',
                            'average_clustering_coefficient_dot_users',
                            'average_clustering_coefficient_dot_items',
                            'average_clustering_coefficient_min_users',
                            'average_clustering_coefficient_min_items',
                            'average_clustering_coefficient_max_users',
                            'average_clustering_coefficient_max_items',
                            'average_degree_log',
                            'average_degree_users_log',
                            'average_degree_items_log',
                            'average_clustering_coefficient_dot_log',
                            'average_clustering_coefficient_min_log',
                            'average_clustering_coefficient_max_log',
                            'average_clustering_coefficient_dot_users_log',
                            'average_clustering_coefficient_dot_items_log',
                            'average_clustering_coefficient_min_users_log',
                            'average_clustering_coefficient_min_items_log',
                            'average_clustering_coefficient_max_users_log',
                            'average_clustering_coefficient_max_items_log',
                            'degree_assortativity_users',
                            'degree_assortativity_items']
