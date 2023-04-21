import datetime
import pandas as pd
import numpy as np 
import sys
import os



today=int(datetime.datetime.now().strftime('%Y%m%d'))
updated_df1 = pd.read_csv('/home/lzx/Daily_data_process/updated_cffex_info.csv', index_col=0, sep='\t')
updated_df2 = pd.read_csv('/home/lzx/Daily_data_process/updated_futures_info.csv', index_col=0, sep='\t')

wrong_num=0

for index_i, row_i in updated_df1.iterrows():
    print(row_i)
    sys.exit()
    file_name = '/shared/shared_data/cffex_data/daily/'+index_i.split('.')[0]+'.csv'
    temp_data = pd.read_csv(file_name, index_col=0, sep='\t')
    temp_data.drop('settle', axis=1, inplace=True)
    if today not in temp_data.index.tolist():
        print(file_name+' dont have this day')
        wrong_num =10
    else:
        temp_data = temp_data.loc[today]
        res = pd.isna(temp_data).sum()
        if res>=2:
            print(file_name + ' too much nan')
            wrong_num+=1 
        res = (temp_data==0).sum()
        if res>=4:
            print(file_name + ' too much zero')
            wrong_num+=1


for index_i, row_i in updated_df2.iterrows():
    if row_i.exg!='XZCE':
        contract= index_i.split('.')[0].lower()
    file_name = '/shared/shared_data/comdty_data/daily/'+contract+'.csv'
    temp_data = pd.read_csv(file_name, index_col=0, sep='\t')
    temp_data.drop('settle', axis=1, inplace=True)
    if today not in temp_data.index.tolist():
        print(file_name+' dont have this day')
        wrong_num = 10
    else:
        temp_data = temp_data.loc[today]
        res = pd.isna(temp_data).sum()
        if res>=2:
            print(file_name + ' too much nan')
            wrong_num+=1
        res = (temp_data==0).sum()
        if res>=4:
            print(file_name + ' too much zero')
            wrong_num+=1 


if wrong_num>10:
    print('\n'*20)
    print('*'*20)
    print('re-updateding')
    print('*'*20)
    os.system('/home/lzx/anaconda3/envs/my_py_env/bin/python3.8 /shared/strategy_data/JK_daily_update.py')
    
# test for git
