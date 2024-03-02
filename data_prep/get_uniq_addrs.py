# subset nodes based on their activity frequency to get institutional/algorithmic entities
import os,sys
import numpy as np
import pandas as pd

#token targets
tokens = ['WBTC', 'WETH', 'DAI', "USDC", 'USDT']

for tk in tokens:
    # target data 
    token_df = pd.read_csv(f'../import/top5/import/{tk}.csv')

    # get time spans 
    min_bl = token_df['blockNumber'].min()
    max_bl = token_df['blockNumber'].max()
    tot_dur = max_bl - min_bl
    print(f'total duration (blocks) : {tot_dur}')


    # binning of transactions 

    # scenario 1. hourly basis
    bls_for_binning = 7200
    n_chunks = np.floor(tot_dur/bls_for_binning) # we will only consider time intervals which are full in duration.


    curr_st_bl = min_bl
    for ii in np.arange(n_chunks):
        upper_bound = curr_st_bl + bls_for_binning
        chunk_df = token_df[token_df['blockNumber'].between(curr_st_bl, upper_bound)]
        curr_st_bl = upper_bound

        if (ii==0):
            all_addrs = pd.concat([chunk_df['from'], chunk_df['to']])
            uniq_addrs = all_addrs.unique()
        else:
            all_addrs = pd.concat([chunk_df['from'], chunk_df['to']])
            all_addrs = all_addrs.unique()
            uniq_addrs = np.intersect1d(uniq_addrs, all_addrs)

    # handle some edges cases (addresse used for burning tokens)
    uniq_addrs = uniq_addrs[uniq_addrs!='0x000000000000000000000000000000000000dead']
    uniq_addrs = uniq_addrs[uniq_addrs!='0x0000000000000000000000000000000000000000']


    # get resulting uniq_addrs
    print(f'number of unique addresses for {tk} : {len(uniq_addrs)}')

    save_dir = '../scratch/uniq_addrs'

    if(not(os.path.exists(save_dir))):
        os.makedirs(save_dir)
    
    # save unique addresses
    np.save(f"{save_dir}/{tk}", uniq_addrs)


