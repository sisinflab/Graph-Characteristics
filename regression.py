import pandas as pd
import numpy as np
import argparse
import statsmodels.formula.api as sm

np.random.seed(1234)

parser = argparse.ArgumentParser(description="Run regression.")
parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--end_id', type=int, default=99)
parser.add_argument('--characteristics', type=str, default='space_size_log shape_log density_log gini_item gini_user')
parser.add_argument('--metric', type=str, default='recall')
args = parser.parse_args()

results = pd.read_csv(f'data/{args.dataset}/characteristics_{args.metric}_{args.start_id}_{args.end_id}.tsv', sep='\t')
models = ['LightGCN', 'DGCF', 'UltraGCN', 'SVDGCN']
characteristics = args.characteristics.split(' ')
results[characteristics] = results[characteristics].apply(
        lambda x: (x - x.mean()))
msk = np.random.rand(len(results)) < 0.8
test = results[~msk]
train = results[msk]

models_results = []

for idx, m in enumerate(models):
    X = train[characteristics]
    y = train[m]

    formula_str_ml = y.name + ' ~ ' + '+'.join(characteristics)
    model_ml = sm.ols(formula=formula_str_ml, data=train[characteristics+[m]])
    fitted_ml = model_ml.fit(cov_type='HC1')
    # predicts_ml = fitted_ml.predict(test)

    # mean_absolute_error_ml = mean_absolute_error(test[characteristics],
    #                                                  predicts_ml)
    # root_mean_absolute_error_ml = math.sqrt(
    #     mean_absolute_error(test[characteristics], predicts_ml))
    # rmse_ml = mean_squared_error(test[characteristics], predicts_ml)

    models_results.append({
        'model': m,
        'score': fitted_ml.rsquared,
        'adjusted_score': fitted_ml.rsquared_adj,
        **fitted_ml.params.to_dict(),
        **fitted_ml.pvalues.rename(lambda x: 'p_'+x).to_dict()
    })

df = pd.DataFrame.from_dict(models_results)
df.to_csv(f'data/{args.dataset}/regression-{args.metric}-{args.start_id}-{args.end_id}.tsv', sep='\t', index=None)
