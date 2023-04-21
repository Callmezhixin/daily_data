'''
author ---- zhixin liu
return ---- download the daily data by JK
'''

import pandas as pd
import numpy as np
import jqdatasdk as jqd
import warnings
import datetime
import logging
import json
import sys
import os
import re

with open('/shared/database/symbol_info','r') as f:
    temp_sym_info = json.load(f)

sym_info_df = pd.DataFrame(temp_sym_info)
multiplier = sym_info_df.loc['multiplier']
multiplier['SI']=5
tick_size = sym_info_df.loc['ticksize']
tick_size['SI']=5
multiplier.rename(lambda x:x.upper(), inplace=True)
tick_size.rename(lambda x:x.upper(), inplace=True)

today_date_str = datetime.datetime.now().strftime('%Y%m%d')
def set_dominant_contract():
    '''
    :return: find the dominant and subdominant contract on each trading day
    '''
    # calendar_df=pd.read_csv(self.path_sr.calendar_path, sep='\t')
    cffex_path_temp = '/shared/shared_data/cffex_data/daily/'
    daily_files = os.listdir(cffex_path_temp)
    daily_files = [x.split('.')[0] for x in daily_files if 'dominant_contract' not in x]
    symbols_ls = np.unique([re.findall(r'([A-Za-z]+)', x) for x in daily_files])
    all_domi_df = pd.DataFrame()
    for i, each_symbol in enumerate(symbols_ls):
        print(each_symbol + ' begins')
        symbol_files = [x for x in daily_files if each_symbol == re.findall(r'([A-Za-z]+)', x)[0]]
        symbol_volume_df = pd.DataFrame()
        for each_data in symbol_files:
            temp_data = pd.read_csv(cffex_path_temp + each_data + '.csv', sep='\t', index_col=0)
            if temp_data.empty:
                continue
            temp_vol = temp_data[['volume']].copy()
            temp_vol.columns = [temp_data.contract.iloc[0]]
            symbol_volume_df = pd.concat([symbol_volume_df, temp_vol], axis=1)
        res = symbol_volume_df.rank(axis=1, ascending=False, method='first')
        res_rank = res.apply(lambda x: pd.Series({v: k for k, v in x.items() if v > 0}), axis=1)
        res_rank = res_rank.iloc[:, [0, 1]]
        res_rank.columns = pd.MultiIndex.from_product([[each_symbol], ['dominant', 'sub_domi']],
                                                      names=['symbol', 'type'])
        all_domi_df = pd.concat([all_domi_df, res_rank], axis=1)
        print(str(len(symbols_ls) - i) + ' left\n')
    all_domi_df.sort_index(ascending=True, inplace=True)
    all_domi_df.to_csv(cffex_path_temp + 'dominant_contract.csv', sep='\t')


def update_dominant_contract():
    cffex_path_temp = '/shared/shared_data/cffex_data/daily/'
    if not os.path.exists(cffex_path_temp + 'dominant_contract.csv'):
        dc_period = pd.DataFrame()
    else:
        dc_period = pd.read_csv(cffex_path_temp + 'dominant_contract.csv', index_col=0, header=[0, 1], sep='\t')
    updated_futures_df = pd.read_csv('/home/lzx/Daily_data_process/updated_cffex_info.csv', sep='\t',
                                     index_col=0)
    updated_sym_ls = updated_futures_df['name'].unique().tolist()
    daily_files = os.listdir(cffex_path_temp)
    daily_files = [x.split('.')[0] for x in daily_files if 'dominant_contract' not in x]
    daily_files = [x for x in daily_files if x in updated_sym_ls]
    updated_sym_ls = [re.findall(r'([A-Za-z]+)', x)[0] for x in updated_sym_ls]
    updated_sym_ls = np.unique(updated_sym_ls)
    all_domi_df = pd.DataFrame()
    for i, each_symbol in enumerate(updated_sym_ls):
        print(each_symbol + ' begins')
        symbol_files = [x for x in daily_files if each_symbol == re.findall(r'([A-Za-z]+)', x)[0]]
        symbol_volume_df = pd.DataFrame()
        for each_data in symbol_files:
            temp_data = pd.read_csv(cffex_path_temp + each_data + '.csv', sep='\t', index_col=0)
            if temp_data.empty:
                continue
            temp_vol = temp_data[['volume']].copy()
            temp_vol.columns = [temp_data.contract.iloc[0]]
            symbol_volume_df = pd.concat([symbol_volume_df, temp_vol], axis=1)
        res = symbol_volume_df.rank(axis=1, ascending=False, method='first')
        res_rank = res.apply(lambda x: pd.Series({v: k for k, v in x.items() if v > 0}), axis=1)
        res_rank = res_rank.iloc[:, [0, 1]]
        res_rank.columns = pd.MultiIndex.from_product([[each_symbol], ['dominant', 'sub_domi']],
                                                      names=['symbol', 'type'])
        all_domi_df = pd.concat([all_domi_df, res_rank], axis=1)
        print(str(len(updated_sym_ls) - i) + ' left\n')
    
    all_domi_df.sort_index(ascending=True, inplace=True)
    all_domi_df = all_domi_df[all_domi_df.index>dc_period.index.max()]
    all_domi_df = pd.concat([dc_period, all_domi_df], axis=0)
    all_domi_df = all_domi_df[~all_domi_df.index.duplicated(keep='first')]
    all_domi_df.to_csv(cffex_path_temp + 'dominant_contract.csv', sep='\t')


warnings.filterwarnings('ignore')
logging.basicConfig(filename='/home/lzx/logs_check/daily.log',
                    format='[%(asctime)s] %(levelname)s %(filename)s %(message)s',
                    level=logging.INFO)
jqd.auth('17621627675', 'Zytz2020')
# jqd.auth('18918605795', '123456Ha')

res_path = '/shared/shared_data/comdty_data/daily/'
today_str = datetime.datetime.today().strftime('%F')
all_futures_info_df = jqd.get_all_securities(types=['futures'], date=None)
all_futures_info_df.index.name = 'contract'
part_futures_info_df = all_futures_info_df[all_futures_info_df.end_date >= today_str].copy()
part_futures_info_df = part_futures_info_df[part_futures_info_df.start_date <= today_str].copy()
part_futures_info_df['exg'] = part_futures_info_df.index.map(lambda x: x.split('.')[1])
part_futures_info_df = part_futures_info_df[part_futures_info_df.exg != 'CCFX']
part_futures_info_df = part_futures_info_df[
    part_futures_info_df.index.map(lambda x: ('8888' not in x) and ('9999' not in x))]
part_futures_info_df['this_start'] = ''

for index_i, row_i in part_futures_info_df.iterrows():
    old_data_path = res_path + row_i.loc['name'] + '.csv'
    if not os.path.exists(old_data_path):
        old_data = pd.DataFrame()
    else:
        old_data = pd.read_csv(old_data_path, sep='\t', index_col=0)
    if not old_data.empty:
        start_date_str = str(old_data.index[-1])
        start_date_str = datetime.datetime.strptime(start_date_str, '%Y%m%d').strftime('%F')
    else:
        start_date_str = row_i.loc['start_date']

    part_futures_info_df.loc[index_i, 'this_start'] = start_date_str
    data = jqd.get_price(index_i, start_date=start_date_str, end_date=today_str, frequency='daily',
                         fields=['open', 'high', 'low', 'close', 'volume', 'money', 'pre_close', 'open_interest'],
                         skip_paused=True, fq=None)
    data.index.name = 'date'
    #data = data[data.index>start_date_str]
    symbol = re.findall('([A-Za-z]+)',index_i)[0]
    temp_multi = multiplier.loc[symbol]
    temp_tick = tick_size.loc[symbol]
    data = data[data.index==today_date_str]
    data.rename(columns={'money': 'amount', 'open_interest': 'oi'}, inplace=True)
    data.rename(lambda x: int(x.strftime('%Y%m%d')), inplace=True)
    data['settle']=((data.amount/data.volume/temp_multi)//temp_tick) * temp_tick
    contract_name = row_i.loc['name']
    data['contract'] = contract_name
    data = pd.concat([old_data, data], axis=0)
    data = data[~data.index.duplicated(keep='last')]
    data.to_csv(res_path + contract_name + '.csv', sep='\t')
    print(contract_name + ' is updated')

part_futures_info_df.to_csv('/home/lzx/Daily_data_process/updated_futures_info.csv', sep='\t')

sys.path.append('/home/lzx/Daily_data_process/')
import chart_pair2

obj = chart_pair2.DataInfo()
obj.update_dominant_contract()

# cffex data update
cffex_path = '/shared/shared_data/cffex_data/daily/'
part_futures_info_df = all_futures_info_df[all_futures_info_df.end_date >= today_str].copy()
part_futures_info_df = part_futures_info_df[part_futures_info_df.start_date <= today_str].copy()
part_futures_info_df['exg'] = part_futures_info_df.index.map(lambda x: x.split('.')[1])
part_futures_info_df = part_futures_info_df[part_futures_info_df.exg == 'CCFX']
part_futures_info_df = part_futures_info_df[
    part_futures_info_df.index.map(lambda x: ('8888' not in x) and ('9999' not in x))]
part_futures_info_df['this_start'] = ''

for index_i, row_i in part_futures_info_df.iterrows():
    old_data_path = cffex_path + row_i.loc['name'] + '.csv'
    if not os.path.exists(old_data_path):
        old_data = pd.DataFrame()
    else:
        old_data = pd.read_csv(old_data_path, sep='\t', index_col=0)
    if not old_data.empty:
        start_date_str = str(old_data.index[-1])
        start_date_str = datetime.datetime.strptime(start_date_str, '%Y%m%d').strftime('%F')
    else:
        start_date_str = row_i.loc['start_date']

    part_futures_info_df.loc[index_i, 'this_start'] = start_date_str
    data = jqd.get_price(index_i, start_date=start_date_str, end_date=today_str, frequency='daily',
                         fields=['open', 'high', 'low', 'close', 'volume', 'money', 'pre_close', 'open_interest'],
                         skip_paused=True, fq=None)
    data.index.name = 'date'
    #symbol = re.findall('([A-Za-z]+)',index_i)[0]
    #temp_multi = multiplier.loc[symbol]
    #temp_tick = tick_size.loc[symbol]
    #data = data[data.index>start_date_str]
    data = data[data.index==today_date_str]
    data.rename(columns={'money': 'amount', 'open_interest': 'oi'}, inplace=True)
    data.rename(lambda x: int(x.strftime('%Y%m%d')), inplace=True)
    #data['settle']=((data.amount/data.volume/temp_multi)//temp_tick) * temp_tick
    contract_name = row_i.loc['name']
    data['contract'] = contract_name
    data = pd.concat([old_data, data], axis=0)
    data = data[~data.index.duplicated(keep='last')]
    data.to_csv(cffex_path + contract_name + '.csv', sep='\t')
    print(contract_name + ' is updated')
part_futures_info_df.to_csv('/home/lzx/Daily_data_process/updated_cffex_info.csv', sep='\t')

logging.info(' daily data updated successfully')

set_dominant_contract()
update_dominant_contract()

os.system('chmod 777 -R '+cffex_path)
os.system('chmod 777 -R '+res_path)
