import pandas as pd
import argparse
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

parser = argparse.ArgumentParser(description="Run generate feature importance.")
parser.add_argument('--filter_FAM', type=bool, default=False)
parser.add_argument('--filter_LOG', type=bool, default=False)

if __name__ == '__main__':
    args = parser.parse_args()

    dataset = pd.read_csv("/Users/claudio/Downloads/characteristics_0_600.tsv", sep='\t')
    dataset.drop('idx', axis=1, inplace=True)

    if args.filter_FAM:
        dataset.drop(['space_size_log', 'shape_log', 'density_log', 'gini_item', 'gini_user'], axis=1, inplace=True)
    if args.filter_LOG:
        dataset.drop(['average_degree_log', 'average_degree_users_log', 'average_degree_items_log',
                      'average_clustering_coefficient_dot_log', 'average_clustering_coefficient_min_log',
                      'average_clustering_coefficient_max_log', 'average_clustering_coefficient_dot_users_log',
                      'average_clustering_coefficient_dot_items_log', 'average_clustering_coefficient_min_users_log',
                      'average_clustering_coefficient_min_items_log', 'average_clustering_coefficient_max_users_log',
                      'average_clustering_coefficient_max_items_log'], axis=1, inplace=True)

    features = dataset.columns[:-12].to_list()
    models = dataset.columns[-12:].to_list()
    for model in models:
        partial = dataset[features + [model]]
        X_train, X_test, y_train, y_test = train_test_split(partial[features].values, partial[model].values,
                                                            test_size=0.2, random_state=12)
        xgb = XGBRegressor(n_estimators=100)
        xgb.fit(X_train, y_train)
        perm_importance = permutation_importance(xgb, X_test, y_test)
        sorted_idx = perm_importance.importances_mean.argsort()
        print(dict(zip(partial.columns[sorted_idx], xgb.feature_importances_[sorted_idx])))