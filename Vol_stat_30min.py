'''
author ---- zhixin liu
return ---- calculate mean/std of last 100/500 |H-L|/O based on 30min bar 

'''
import pandas as pd
import numpy as np
import logging
import datetime
import json
import sys
import gc
import os

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)
logging.basicConfig(filename='/home/lzx/logs_check/daily.log',
                    format='[%(asctime)s] %(levelname)s %(filename)s %(message)s',
                    level=logging.INFO)

datenum= int(datetime.datetime.now().strftime('%Y%m%d'))
money = 200000
path_dict = {'calendar_path': '/shared/database/shfe_trading.calendar',
             'symbol_info_path': '/shared/database/Product_info.json'}

path_sr = pd.Series(path_dict)
calendar_df = pd.read_csv(path_sr.calendar_path, sep='\t', index_col=0)
exchanges = ['zce', 'dce', 'shfe', 'cffex']
#exchanges=['cffex']
margin_df = pd.read_csv('/home/lzx/sync_signals/margins.csv', index_col=1)
margin_df = margin_df[pd.notnull(margin_df.index)]
# margin_df.dropna(how='any', axis=0, inplace=True)
# margin_df.loc['rb']
with open(path_sr.symbol_info_path, 'r') as f:
    symbol_info_df = pd.DataFrame(json.load(f))
multiple_df = symbol_info_df.loc['multiple']
multiple_df['IM']=200
margin_df.loc['IM','margin_rate']=15

res_path = '/shared/strategy_stats/volatility_stats_30mins.csv'


# res_path = 'volatility_stats_30mins.csv'

def eod_stats(temp_df, date_num):
    '''

    :param temp_df:
    :param date_num:
    :return: each day's statistics
    '''
    df = temp_df.copy()
    df.reset_index(inplace=True)
    df.set_index('time', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    temp_symbol = temp_df.symbol.unique()[0]


    if temp_symbol in ['T', 'TS', 'TF']:
        date_eod = datetime.datetime.strptime(str(date_num), '%Y%m%d').strftime('%F') + ' 15:15:00'
    else:
        date_eod = datetime.datetime.strptime(str(date_num), '%Y%m%d').strftime('%F') + ' 15:00:00'
    try:
        date_loc = df.index.get_loc(date_eod)
        period_100_mean = df.iloc[max(date_loc - 99, 0):date_loc + 1]['|(H-L)/O|%'].mean(skipna=False) * 100
        period_500_mean = df.iloc[max(date_loc - 499, 0):date_loc + 1]['|(H-L)/O|%'].mean(skipna=False) * 100
        period_100_std = df.iloc[max(date_loc - 99, 0):date_loc + 1]['|(H-L)/O|%'].std(skipna=False) * 100
        period_500_std = df.iloc[max(date_loc - 499, 0):date_loc + 1]['|(H-L)/O|%'].std(skipna=False) * 100
        num2buy = max(1, round(money / df.iloc[date_loc].close / multiple_df[df.iloc[date_loc].symbol]))
        stoploss1 = min(period_100_mean + 2.5 * period_100_std, 3)
        stoploss2 = min(period_500_mean + 2.5 * period_500_std, 3)
        market_value = num2buy * df.iloc[date_loc].close * multiple_df[df.iloc[date_loc].symbol]
        moneyloss1 = market_value * stoploss1 / 100
        moneyloss2 = market_value * stoploss2 / 100
        margin = market_value * margin_df.loc[df.iloc[date_loc].symbol, 'margin_rate'] / 100
        res = pd.DataFrame([[period_100_mean, period_500_mean, period_100_std, period_500_std,
                             stoploss1, moneyloss1, stoploss2, moneyloss2, margin]], index=[date_eod])
        res_col = pd.MultiIndex.from_product(
            [[df.symbol.iloc[0]], ['100mean', '500mean', '100std', '500std',
                                   'stoploss1', 'moneyloss1', 'stoploss2', 'moneyloss2', 'margin']])
        res.columns = res_col
        if temp_symbol in ['T', 'TS', 'TF']:
            res.index = res.index.str.replace('15:15:00', '15:00:00')
        return res.T
    except:
        print(temp_symbol + ' not calculated')


# *************************************************** start
if not os.path.exists(res_path):
    update_dates_ls = (calendar_df.index[calendar_df.index >= 20170101]).tolist()
    res_df = pd.DataFrame()
else:
    res_df = pd.read_csv(res_path, index_col=[0, 1], sep='\t')
    res_df.index.names = (None, None)
    res_df = res_df.unstack()
    res_df.columns = res_df.columns.swaplevel()
    update_dates_ls = (calendar_df.index[calendar_df.index >= res_df.index[-1]]).tolist()
    if len(update_dates_ls) == 0:
        logging.info(datetime.datetime.today().strftime(
            '%F') + ' vol not updated, latest day based on calendar file is ' + str(res_df.index[-1]))

if len(update_dates_ls) > 0:
    res_each_exch = pd.DataFrame()
    for exch in exchanges:
        # load data, combine day and night

        convert_day_path = '/shared/shared_data/convert_data.{}.{}'.format(exch, 'day')
        convert_day_data = pd.read_csv(convert_day_path + '.csv', sep='\t', index_col=0)
        if exch == 'cffex':
            convert_night_data = pd.DataFrame()
        else:
            convert_night_path = '/shared/shared_data/convert_data.{}.{}'.format(exch, 'night')
            convert_night_data = pd.read_csv(convert_night_path + '.csv', sep='\t', index_col=0)
        convert_data = pd.concat([convert_day_data, convert_night_data], axis=0)
        if datenum not in convert_data.index:
            if exch == 'cffex':
                convert_day_path = '/shared/shared_data/convert_jq.{}.{}'.format(exch, 'day')
                convert_day_data = pd.read_csv(convert_day_path + '.csv', sep='\t', index_col=0)
                convert_night_path = '/shared/shared_data/convert_jq.{}.{}'.format(exch, 'night')
                convert_night_data = pd.read_csv(convert_night_path + '.csv', sep='\t', index_col=0)
                convert_data = pd.concat([convert_day_data, convert_night_data], axis=0)
            else:
                convert_data = pd.read_csv('shared/shared_data/cffex_data/cffex.csv', sep='\t')
                convert_data.drop('time', axis=1, inplace=True)
                convert_data.rename(columns={'timestamp_exchange':'time'},inplace=True)


        # filter dominnat data
        convert_data = convert_data[convert_data.symbol.map(lambda x: '1' not in x)]
        print(convert_data.symbol.unique())
        convert_data['|(H-L)/O|%'] = (np.abs(convert_data.high - convert_data.low) / convert_data.open)
        res_each_date = pd.DataFrame()
        for date_int in update_dates_ls:
            each_exch_res = convert_data.groupby('symbol', as_index=False).apply(eod_stats, date_int)
            each_exch_res.index = each_exch_res.index.droplevel(level=0)
            each_exch_res = each_exch_res.T
            res_each_date = pd.concat([res_each_date, each_exch_res], axis=0) 
            print('{} {} is done'.format(exch, date_int))
        res_each_exch = pd.concat([res_each_exch, res_each_date], axis=1)

    res_each_exch.rename(lambda x: int(datetime.datetime.strptime(x[:10], '%Y-%m-%d').strftime('%Y%m%d')), inplace=True)
    res_df = pd.concat([res_df, res_each_exch])
    res_df = res_df[~res_df.index.duplicated(keep='last')]
    res_df.sort_index(ascending=False, inplace=True)
    res_df = res_df.stack(level=0)
    res_df.index.names = ('date', 'symbol')
    #res_df.to_csv(res_path, float_format='%.3f', sep='\t')

    logging.info(' vol updated successfully')
