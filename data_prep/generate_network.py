# generate networks based on different assumptions

import os,sys
import numpy as np
import pandas as pd
import networkit as nk

#token targets
tokens = ['WBTC', 'WETH', 'DAI', "USDC", 'USDT']

token = 'WBTC'

# data locations
uniq_addrs = np.load(f'../scratch/uniq_addrs/{token}.npy', allow_pickle=True)
token_df = pd.read_csv(f'../import/top5/import/{token}.csv')


# bin transfer activity using different time windows
min_bl = token_df['blockNumber'].min()
max_bl = token_df['blockNumber'].max()
tot_dur = max_bl - min_bl

bin_size = 7200 # number of blocks
n_chunks = np.floor(tot_dur/bin_size) # we will only consider time intervals which are full in duration.

# subset for only those nodes appear in uniq_addrs
token_df = token_df[token_df['from'].isin(uniq_addrs)|token_df['to'].isin(uniq_addrs)]


df_list=[]
curr_st_bl = min_bl
for ii in np.arange(n_chunks):
    upper_bound = curr_st_bl + bin_size
    chunk_df = token_df[token_df['blockNumber'].between(curr_st_bl, upper_bound)]
    curr_st_bl = upper_bound
    df_list.append(chunk_df)





# scenario 1. consider real transfer transactions as edge weights for system dynamics matrix (A)

nk_list = [] # directed, weighted

for chunk_num, data_df in enumerate(df_list):
    # subset only necessary columns
    data_df = data_df[['from', 'to', 'value']]

    # take out edge cases 
    special_addrs = ['0x000000000000000000000000000000000000dead', '0x0000000000000000000000000000000000000000']

    for addr in special_addrs:
        data_df = data_df[data_df['from']!=addr]
        data_df = data_df[data_df['to']!=addr]

    # get edge weights
    grouped = data_df.groupby(['from', 'to']).agg(['sum', 'mean']).reset_index()
    grouped['from_idx'] = grouped['from'].apply(lambda x: list(uniq_addrs).index(x)) # there shouldn't be any error
    grouped['to_idx'] = grouped['to'].apply(lambda x: list(uniq_addrs).index(x)) # there shouldn't be any error

    # Create an empty directed graph
    G = nk.graph.Graph(weighted=True, directed=True)

    # Add nodes; note that node ids in NetworKit are 0-based integers and it doesn't take string as identifiers
    for ii in np.arange(len(uniq_addrs)):
        G.addNode()
  

    # Add directed weighted edges: G.addEdge(source, target, weight)
    for ii,row in grouped.iterrows():
        G.addEdge(row['from_idx'], row['to_idx'], row['value']['sum']) # Edge from node 0 to 1 with weight 2.0
    
    nk_list.append(G)
    print(f'networkit graph was successfully created for the bin size : {bin_size} blocks / chunk: {chunk_num+1}/{len(df_list)}')



# scenario 2. compute node balances changes and derive dynamics matrix (A)(need a long observation time => inversion of the dynamics matrix A)







# general profiling
# import warnings
# warnings.filterwarnings('ignore')

config = nk.profiling.Config.createConfig(preset='minimal')
nk.profiling.Profile.create(nk_list[0], config=config).show() # need to use subsets otherwise takes too long



# convert graphs into unweighted graphs for graph metrics 














# plotting 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')    
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600



# save to graphML