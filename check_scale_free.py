import pandas as pd
import networkx
import powerlaw
import math
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt


dataset = 'gowalla'
data = pd.read_csv(f'./data/{dataset}/dataset.tsv', sep='\t', header=None)
_public_to_private_users = {el: idx for idx, el in enumerate(data[0].unique().tolist())}
n_users = data[0].nunique()
_public_to_private_items = {el: idx + n_users for idx, el in enumerate(data[1].unique().tolist())}
data[0] = data[0].map(_public_to_private_users)
data[1] = data[1].map(_public_to_private_items)
graph = networkx.Graph()
graph.add_nodes_from(list(range(data[0].nunique())), bipartite='users')
graph.add_nodes_from(list(range(data[1].nunique())), bipartite='items')
graph.add_edges_from(list(zip(data[0], data[1])))
if not networkx.is_connected(graph):
    graph = graph.subgraph(max(networkx.connected_components(graph), key=len))
degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
fit = powerlaw.Fit(degree_sequence, xmin=1)
R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
print(R, p)
count_degree = dict(Counter(degree_sequence))
count_degree = dict(OrderedDict(sorted(count_degree.items(), reverse=True)))
count_degree_plot = {key: value / sum(count_degree.values()) for key, value in count_degree.items()}
power_law_plot = {key: math.pow(key, -fit.alpha) for key, value in count_degree.items()}
# fig2 = fit.plot_pdf(color='b', linewidth=2)

# plt.style.use("ggplot")
import tikzplotlib

plt.scatter(y=list(count_degree_plot.values()), x=list(count_degree_plot.keys()))
# plt.plot(list(power_law_plot.keys()), list(power_law_plot.values()), '--', color='green')
fit.power_law.plot_pdf(color='g')
fit.exponential.plot_pdf(color='r')
plt.xscale("log")
plt.yscale("log")
plt.ylim(bottom=10e-6)
plt.ylim(top=10e-1)
# x, y = fit.power_law.plot_pdf(color='g', linestyle='--', ax=fig2)
# plt.plot(list(power_law_plot.keys()), list(power_law_plot.values()), '--', color='green')
plt.show()
tikzplotlib.save(f'./data/{dataset}/powerlaw.tex')
