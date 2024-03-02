%load_ext autoreload
%autoreload 2

# generate networks based on different assumptions
import os,sys
import numpy as np
import pandas as pd
import networkit as nk
import networkx as nx

sys.path.append('../')

from BNC import BNC # util class for the current BNC work

# result dict 

result={}
comp_dist = {}
#token targets
tokens = ['WBTC', 'WETH', 'DAI', "USDC", 'USDT']

# concatenate all of them together
concat_addrs = []
concat_tk_df = pd.DataFrame()

for token in tokens:
    # token = 'WBTC'
    result['token'] = token
    # data locations
    uniq_addrs = np.load(f'../scratch/uniq_addrs/{token}.npy', allow_pickle=True)
    token_df = pd.read_csv(f'../import/top5/import/{token}.csv')

    concat_addrs.append(uniq_addrs)
    concat_tk_df = pd.concat([concat_tk_df, token_df])
uniq_addrs = np.concatenate(concat_addrs, 0)

# unique nodes
uniq_addrs = np.unique(uniq_addrs)

#all token transfers 
token_df = concat_tk_df

# init BNC class (gets basic stats on analysis setting)
bnc = BNC(all_addrs = uniq_addrs, token_df = token_df)

# subset for only those nodes appear in uniq_addrs
token_df = bnc.get_relevant_transfers(token_df)

# apply simple conversion of usd prices 
price_table = {}
price_table['WBTC'] = 60000
price_table['WETH'] = 3000

# scenario 1. consider real transfer transactions as edge weights for the system dynamics matrix (A)
bnc.set_pricetable(price_table)
token_df = bnc.lookup_compute_price(token_df)


# subset only necessary columns
data_df = token_df[['from', 'to', 'value']]


# get edge weights
grouped = bnc.compute_edge_weights(data_df, agg_methods=['sum'])
grouped = bnc.assign_edge_indices(grouped)

# create graph 
G = bnc.generate_dir_weighted_graph(grouped, agg_method= 'sum')


# remove self-loops
bnc.remove_self_loop(G)


# get components and get the largest
scc_result = bnc.get_scc(G)
lc_result = bnc.get_largest_comp(G)


# plot degree / componenet distributions
# nk.plot.degreeDistribution(G)
# nk.plot.connectedComponentsSizes(und_G)

# plot degree distribution of the largest component 

# get the largest comp
# intialize new bnc class using the only nodes belonging to the largest component
bnc_lc = BNC(all_addrs = lc_result['lc_uniq_addrs'], token_df = token_df)

# subset for only those nodes appear in uniq_addrs
token_df = bnc_lc.get_relevant_transfers(token_df) # token_df gets subset for the largest component

# subset only necessary columns
data_df = token_df[['from', 'to', 'value']]


# get edge weights
grouped = bnc_lc.compute_edge_weights(data_df, agg_methods=['sum'])
grouped = bnc_lc.assign_edge_indices(grouped)

# generate weighted digraph
G = bnc_lc.generate_dir_weighted_graph(grouped, agg_method='sum')

# remove self-loops
G.removeSelfLoops()

# to undirected 
und_G = nk.graphtools.toUndirected(G)

# or to unweighted
unw_G = nk.graphtools.toUnweighted(G)

# compute conventional maximum matching from a undirected graph 
# export it to networkx
unmatched_nodes_count = bnc_lc.get_unmatched_nodes_count(unw_G)


# compare it with ER graph surrogates
er_surrogate_result = bnc_lc.get_unmatched_nodes_count_ER(unw_G, iter=100)
er_mean = np.mean(er_surrogate_result)
er_std = np.std(er_surrogate_result)
print(f'ER mean : {er_mean}, ER std : {er_std}')




####################################################################################################

# compute HITS on the directed graph (weight is not considered)
hubs, authorities = bnc_lc.get_hits_nodes(G, max_iter=None, tol=1e-08, nstart=None, normalized=True)



# plot the distribution of hubs / authorities
sorted_hubs_idx = np.argsort(list(hubs.values()))[::-1]
sorted_authorities_idx = np.argsort(list(authorities.values()))[::-1]


sorted_hubs = np.array(list(hubs.values()))[sorted_hubs_idx]
#use the same ordering for authorities
sorted_authorities = np.array(list(authorities.values()))[sorted_hubs_idx]


plt.style.use('ggplot')    
# fig = plt.figure()
ax = fig.add_subplot()
ax.plot(sorted_authorities, label='authority')
ax.plot(sorted_hubs, label='hub')
ax.set_xlabel('Node(id)')
ax.set_ylabel('Score')
plt.legend()


# re-sort the authorities indices
sorted_authorities = np.array(list(authorities.values()))[sorted_authorities_idx]

# get knee point 
knee_p_hubs = bnc_lc.get_knee_point(sorted_hubs, return_plot=True)
knee_p_author = bnc_lc.get_knee_point(sorted_authorities, return_plot=True)



# compute max flow 
# backup_G = G
mf_hubs = bnc_lc.get_maximal_flow(G, sorted_hubs_idx, knee_p_hubs)
mf_author = bnc_lc.get_maximal_flow(G, sorted_hubs_idx, knee_p_hubs)



# get influence (usd) 
inf_hubs_df = bnc_lc.get_influence(mf_hubs)
inf_author_df = bnc_lc.get_influence(mf_author)


# compare this to degree distribution (unweighted)
G_deg = bnc_lc.get_degrees(unw_G)
G_deg = np.array(G_deg)[:,1]

# deg_idx = np.argsort(G_deg)


# Fig.1C scatter plot against deg.
from sklearn import preprocessing
sd_g_deg = preprocessing.scale(G_deg)
sd_hub_inf = preprocessing.scale(inf_hubs_df['influence'])
sd_author_inf = preprocessing.scale(inf_author_df['influence'])

# quickly check linear reg
import scipy 
hub_test = scipy.stats.pearsonr(sd_g_deg, sd_hub_inf, alternative='two-sided')
author_test = scipy.stats.pearsonr(sd_g_deg, sd_hub_inf, alternative='two-sided')


# plotting myself. 
# plotting 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')    
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600


# Fig.1A. plot component distribution
comp_len = list(scc_result['sizes'].values()) 
fig = plt.figure(figsize=(8,3))
ax = fig.add_subplot(131)
sns.violinplot(ax=ax,data=comp_len, color='gray', saturation=0.1)
# sns.catplot(ax=ax,data=comp_len, color='gray', saturation=0.1)
sns.stripplot(ax=ax,data=comp_len, color='darkblue', alpha=0.5)
plt.yscale('log')
ax.set_xlabel('Components')
ax.set_ylabel('Component Size')
ax.set_xticklabels([])



# below is for plotting
nx_graph=nk.nxadapter.nk2nx(backup_G)
und_dx_graph = nx.to_undirected(nx_graph)
tot_nodes = len(list(und_dx_graph.nodes()))
density = nx.density(und_dx_graph)
erg = nk.generators.ErdosRenyiGenerator(tot_nodes, density, directed=False)
ergG = erg.generate()
# nk.plot.degreeDistribution(ergG)

# fig.1B. plot deg dist 
# plt.style.use('default')  
# nk.plot.degreeDistribution(unw_G)
G_deg = bnc_lc.get_degrees(unw_G)
G_deg = np.array(G_deg)[:,1]
logbins = np.geomspace(G_deg.min(), G_deg.max(), 8)
er_deg = bnc_lc.get_degrees(ergG)
er_deg = np.array(er_deg)[:,1]

ax = fig.add_subplot(132)
ax.hist(G_deg, bins=logbins, label='transfer')
ax.hist(er_deg, bins=logbins, label='ER')
plt.xscale('log')
ax.set_xlabel('Degree')
ax.set_ylabel('Count')
ax.legend()


ax = fig.add_subplot(133) 
ax.scatter(sd_g_deg,sd_hub_inf , label=f'hub')
ax.scatter(sd_g_deg,sd_author_inf , label=f'authority')
ax.set_xlabel('Degree (n.u.)')
ax.set_ylabel('Influence (n.u.)')
ax.legend()

plt.tight_layout()