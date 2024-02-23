import os,sys
import boto3
import numpy as np
import pandas as pd
from datetime import datetime
import parmap
# get transfer traces from s3
bucket_name = 'neo4j-upload-transfer-decode'


# starting datetime
min_time = datetime(year=2024, month=2, day=17, hour=4)


# set boto3 client
s3_client = boto3.client('s3')
 
 
# List objects within the bucket
s3_paginator = s3_client.get_paginator('list_objects_v2')
s3_iterator = s3_paginator.paginate(Bucket=bucket_name)
filtered_keys = s3_iterator.search(
    f"Contents[?to_string(LastModified)>'\"{min_time}\"'].Key"
)

# Need to re-initialize it otherwise, s3_iterator only searches the last page
s3_paginator = s3_client.get_paginator('list_objects_v2')
s3_iterator = s3_paginator.paginate(Bucket=bucket_name)
filtered_datetimes = s3_iterator.search(
    f"Contents[?to_string(LastModified)>'\"{min_time}\"'].LastModified"
)

# cast them to lists
csv_files = list(filtered_keys)
datetimes = list(filtered_datetimes)


def get_s3_object(csv_file):
    obj = s3_client.get_object(Bucket=bucket_name, Key=csv_file)

    # Read the object (which is in CSV format) directly into a pandas DataFrame
    s3_csv_name = obj['Body']
    df = pd.read_csv(s3_csv_name)
    return df



# load csv files from the bucket
par_results = parmap.map(get_s3_object, csv_files, pm_pbar=True, pm_processes=8)
par_results = pd.concat(par_results)


# filter the result based on token symbols
target_symbol = ['WETH', 'WBTC', 'DAI', 'USDC', 'USDT']
import_path = '../import'
for target in target_symbol:
    target_df = par_results[par_results['symbol']==target]
    target_df = target_df.sort_values(['blockNumber', 'tx_pos'])
    #save it to the local dir
    target_df.to_csv(f'{import_path}/{target}.csv', index=False)

