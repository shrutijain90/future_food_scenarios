"""
Author: Jasper Verschuur
Edited by: Shruti Jain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data_dir = '../../OPSIS/Data/Trade_clearance_model'

def compare_output(var_old, var_new, bil = True):
    if bil == True:
        base = pd.DataFrame(pd.Series(var_old.extract_values())).reset_index().rename(columns={'level_0':'from_abbreviation','level_1':'to_abbreviation',0:'base'})
        new = pd.DataFrame(pd.Series(var_new.extract_values())).reset_index().rename(columns={'level_0':'from_abbreviation','level_1':'to_abbreviation',0:'new'})
        compare = base.merge(new, on = ['from_abbreviation','to_abbreviation'])
    else:
        base = pd.DataFrame(pd.Series(var_old.extract_values())).reset_index().rename(columns={'index':'abbreviation',0:'base'})
        new = pd.DataFrame(pd.Series(var_new.extract_values())).reset_index().rename(columns={'index':'abbreviation',0:'new'})
        compare = base.merge(new, on = ['abbreviation'])

    compare['diff'] = compare['new'] - compare['base']
    compare['diff_perc'] = np.round(100*(compare['new'] - compare['base']) / compare['base'], 1).replace(np.inf, 100).replace(np.nan, 0)

    return compare

def round_dict(d, k):
    return {key: float(f"{value:.{k}f}") for key, value in d.items()}

def shock(prod_dict, factor):
    return {key: value * factor for key, value in prod_dict.items()}


def dataframe_model_output(var1, var2):
    predicted = pd.DataFrame(pd.Series(var1.extract_values())).reset_index().rename(columns={'level_0':'from_abbreviation','level_1':'to_abbreviation',0:'new'})
    base = pd.DataFrame(pd.Series(var2.extract_values())).reset_index().rename(columns={'level_0':'from_abbreviation','level_1':'to_abbreviation',0:'original'})

    validation = predicted.merge(base, on = ['from_abbreviation','to_abbreviation'])
    return validation


def remove_countries(country, bil_trade):
    
    #### only include countries that either demand crop or supply it ###
    country = country[(country['demand_q']>0) | (country['supply_q']>0)].reset_index(drop = True)

    bil_trade = bil_trade[(bil_trade['from_abbreviation'].isin(country['abbreviation'].unique())) &
                                   (bil_trade['to_abbreviation'].isin(country['abbreviation'].unique()))].reset_index(drop = True)

    return country,  bil_trade


class CountryClass:

    def __init__(self, dataframe):

        ### countries list
        self.abbreviation = list(dataframe['abbreviation'])

        ### production cost origin country
        self.production_cost = dataframe.set_index(['abbreviation'])['prod_price_USD_t']

        ## demand and supply elasticity
        self.demand_elas = dataframe.set_index(['abbreviation'])['demand_elas']*-1 ### make a positive number
        self.supply_elas = dataframe.set_index(['abbreviation'])['supply_elas']

        ### total demand and supply
        self.demand = dataframe.set_index(['abbreviation'])['demand_q']/1000
        self.supply = dataframe.set_index(['abbreviation'])['supply_q']/1000


class BilateralClass:

    def __init__(self, dataframe, factor_error):
        error = 1*10**(-1*factor_error)
        dataframe['q_calib'] = dataframe['q_calib']/1000
        dataframe['q_old'] = dataframe['q_old']/1000

        dataframe['q_calib'] = np.where(dataframe['q_calib']<=error, error, dataframe['q_calib'])
        dataframe['q_old'] = np.where(dataframe['q_old']<=error, error, dataframe['q_old'])

        ### adjust internal trade ###
        dataframe['q_old'] = np.where(((dataframe['from_abbreviation'] == dataframe['to_abbreviation'])&(dataframe['trade_relationship_old']==0)),
                                                        dataframe['q_calib'], dataframe['q_old'])

        dataframe['trade_relationship_old'] = np.where(((dataframe['from_abbreviation'] == dataframe['to_abbreviation'])&(dataframe['trade_relationship']==1)),
                                                        1, dataframe['trade_relationship_old'])

        ## calibration trade
        self.trade01 =  np.round(dataframe.set_index(['from_abbreviation','to_abbreviation'])['q_calib'], factor_error) ## thousand of tonnes

        ## existing trade
        self.trade_old = np.round(dataframe.set_index(['from_abbreviation','to_abbreviation'])['q_old'], factor_error) ## thousand of tonnes

        ### binary for existing trade relation
        self.trade_binary = dataframe.set_index(['from_abbreviation','to_abbreviation'])['trade_relationship_old']

        ### trade cost
        self.tc1 = dataframe.set_index(['from_abbreviation','to_abbreviation'])['trade_USD_t']

        ## ad-valorem tariff
        self.adv = dataframe.set_index(['from_abbreviation','to_abbreviation'])['adv']

def read_model_input(crop_code, error_factor):
    file_country = f'{data_dir}/Input/Country_data/country_information_{crop_code}.csv'
    file_bil = f'{data_dir}/Input/Trade_cost/bilateral_trade_cost_{crop_code}.csv'

    #### read data ###
    country_data = pd.read_csv(file_country)
    bil_trade_data = pd.read_csv(file_bil)
    
    ### remove countries where demand and supply are zero
    country_data, bil_trade_data = remove_countries(country = country_data, bil_trade = bil_trade_data)

    ### create two classes
    country_class = CountryClass(country_data)
    bilateral_class = BilateralClass(bil_trade_data, factor_error = error_factor)

    return country_class, bilateral_class


def compare_trade_flows(model_predict, model_base, error):
    ### predicted trade flows
    predicted_trade = pd.DataFrame(pd.Series(model_predict.extract_values())).reset_index().rename(columns={'level_0':'from_abbreviation','level_1':'to_abbreviation',0:'t_pred'})
    base_trade = pd.DataFrame(pd.Series(model_base.extract_values())).reset_index().rename(columns={'level_0':'from_abbreviation','level_1':'to_abbreviation',0:'t_exist'})

    ### merge
    trade_validation = predicted_trade.merge(base_trade, on = ['from_abbreviation','to_abbreviation'])
    trade_validation['t_pred'] = np.where(trade_validation['t_pred']<error, error, trade_validation['t_pred'])
    trade_validation['error'] = (trade_validation['t_pred']-trade_validation['t_exist'])

    ### binary predicton ###
    trade_validation['correct'] = np.where((trade_validation['t_pred']>1)&(trade_validation['t_exist']>1), 1, 0)

    ### create some statistics  ###
    hit_ratio = trade_validation['correct'].sum()/len(trade_validation[trade_validation['t_pred']>1])
    MSE = mean_squared_error(trade_validation['t_exist'].values,trade_validation['t_pred'].values)
    r2 = r2_score(trade_validation[trade_validation['from_abbreviation']!=trade_validation['to_abbreviation']]['t_exist'].values,trade_validation[trade_validation['from_abbreviation']!=trade_validation['to_abbreviation']]['t_pred'].values)

    print('hit ratio:', np.round(hit_ratio, 2))
    print('MSE:', np.round(MSE, 2))
    print('R-squared:',np.round(r2, 2))


    return trade_validation, hit_ratio, MSE, r2


def scatter_plot_trade(df_output, fname, to_print=False):
    ### plot of trade flows ##
    fig, ax = plt.subplots(figsize = (3.5,3.5))
    plt.scatter(np.log(df_output['t_pred']+1), np.log(df_output['t_exist']+1), s = 8, color = 'grey', alpha = 0.4, zorder = 3)
    plt.plot(np.linspace(0,13, 100), np.linspace(0,13, 100), color = 'k', lw = 0.5, zorder = 1)
    plt.xlim(0, 12); plt.xlabel('Predicted trade ln (x1,000 t)')
    plt.ylim(0, 12); plt.ylabel('Observed trade ln (x1,000 t)')
    if fname is not None:
        plt.savefig(f'{fname}.png', bbox_inches='tight')
    if to_print:
        plt.show()
