import os

DATA_FOLDER = os.path.abspath('./data')
OUTPUT_FOLDER = os.path.abspath('./data/')
RESULT_FOLDER = os.path.abspath('./results/')
ACCEPTED_DATASETS = ['gowalla', 'amazon-book', 'yelp2018']
ACCEPTED_SPLITTINGS = ['edge-dropout', 'node-dropout']
ACCEPTED_METRICS = ['Recall', 'nDCG']
ACCEPTED_CHARACTERISTICS = ['space_size_log', 'shape_log', 'density_log', 'gini_user',
                            'gini_item', 'average_degree_users_log', 'average_degree_items_log',
                            'average_clustering_coefficient_dot_users_log',
                            'average_clustering_coefficient_dot_items_log', 'degree_assortativity_users',
                            'degree_assortativity_items']
