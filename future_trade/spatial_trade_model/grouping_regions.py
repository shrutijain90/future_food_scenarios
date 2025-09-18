# Usage: python -m future_trade.spatial_trade_model.grouping_regions

import pandas as pd
import numpy as np
import os

from future_trade.data_inputs.balance_trade import get_area_codes

def aggregate_elas(demand_elas, supply_elas, dem_sup_all, area_codes):
    # weight demand elasticity by 2020 demand
    # weigh supply elasticity by 2020 supply
    demand_elas = demand_elas.merge(dem_sup_all.drop('supply_q', axis=1)).merge(area_codes).drop('abbreviation', axis=1).rename(
        columns={'abbreviation_new': 'abbreviation'})
    demand_elas.loc[demand_elas['demand_q']==0, 'demand_q'] = 0.01
    supply_elas = supply_elas.merge(dem_sup_all.drop('demand_q', axis=1)).merge(area_codes).drop('abbreviation', axis=1).rename(
        columns={'abbreviation_new': 'abbreviation'})
    supply_elas.loc[supply_elas['supply_q']==0, 'supply_q'] = 0.01
    
    demand_elas['demand_elas'] = demand_elas['demand_elas'] * demand_elas['demand_q']
    demand_elas['demand_elas_baseline'] = demand_elas['demand_elas_baseline'] * demand_elas['demand_q']
    demand_elas = demand_elas.groupby(['IMPACT_code', 'abbreviation', 'year'])[[
        'demand_elas', 'demand_elas_baseline', 'demand_q']].sum().reset_index()
    demand_elas['demand_elas'] = demand_elas['demand_elas'] / demand_elas['demand_q']
    demand_elas['demand_elas_baseline'] = demand_elas['demand_elas_baseline'] / demand_elas['demand_q']
    demand_elas = demand_elas.drop('demand_q', axis=1)
    demand_elas['scaling_factor_demand_elas'] = demand_elas['demand_elas'] / demand_elas['demand_elas_baseline']

    supply_elas['supply_elas'] = supply_elas['supply_elas'] * supply_elas['supply_q']
    supply_elas = supply_elas.groupby(['IMPACT_code', 'abbreviation'])[[
        'supply_elas', 'supply_q']].sum().reset_index()
    supply_elas['supply_elas'] = supply_elas['supply_elas'] / supply_elas['supply_q']
    supply_elas = supply_elas.drop('supply_q', axis=1)

    return demand_elas, supply_elas

def aggregate_bilateral_trade(df_bilateral, df_country, area_codes):
    # weigh trade cost with 2020 flow volume and adv with 2020 approx landing cost share. recalculate q_calib and q_calib old
    df_bilateral = df_bilateral.merge(df_country[['abbreviation', 'prod_price_USD_t']].rename(columns={'abbreviation': 'from_abbreviation'}))
    df_bilateral = df_bilateral.merge(area_codes, left_on='from_abbreviation', right_on='abbreviation').drop(
        ['from_abbreviation', 'abbreviation'], axis=1).rename(columns={'abbreviation_new': 'from_abbreviation'}).merge(
            area_codes, left_on='to_abbreviation', right_on='abbreviation').drop(
                ['to_abbreviation', 'abbreviation'], axis=1).rename(columns={'abbreviation_new': 'to_abbreviation'})

    df_bilateral['flow_dup'] = df_bilateral['q_calib']
    df_bilateral.loc[df_bilateral['flow_dup']==0, 'flow_dup'] = 0.01 # to avoid getting nulls when aggregating 
    df_bilateral['total_trade_cost'] = df_bilateral['trade_USD_t'] * df_bilateral['flow_dup']    
    df_bilateral['tariff_value'] = (df_bilateral['prod_price_USD_t'] + df_bilateral['trade_USD_t']) * df_bilateral['flow_dup']
    df_bilateral.loc[df_bilateral['tariff_value']==0, 'tariff_value'] = 0.0001 # to avoid getting nulls when aggregating 
    df_bilateral['total_tariff'] = (df_bilateral['prod_price_USD_t'] + df_bilateral['trade_USD_t']) * df_bilateral['adv'] * df_bilateral['flow_dup']
    df_bilateral = df_bilateral.groupby(['from_abbreviation', 'to_abbreviation', 'IMPACT_code'])[[
        'q_calib', 'q_old', 'total_trade_cost', 'total_tariff', 'flow_dup', 'tariff_value']].sum().reset_index()
    df_bilateral['trade_USD_t'] = df_bilateral['total_trade_cost'] / df_bilateral['flow_dup']
    df_bilateral['adv'] = df_bilateral['total_tariff'] / df_bilateral['tariff_value']
    df_bilateral = df_bilateral.drop(['total_trade_cost', 'total_tariff', 'flow_dup', 'tariff_value'], axis=1)

    df_bilateral['trade_relationship'] = 0
    df_bilateral.loc[df_bilateral['q_calib']>0, 'trade_relationship'] = 1
    df_bilateral['trade_relationship_old'] = 0
    df_bilateral.loc[df_bilateral['q_old']>0, 'trade_relationship_old'] = 1

    return df_bilateral

def aggregate_demand_scn(demand_scn, df_country, area_codes):
    demand_scn = demand_scn.merge(df_country[['abbreviation', 'demand_q']]).merge(area_codes).drop('abbreviation', axis=1).rename(
        columns={'abbreviation_new': 'abbreviation'})
    demand_scn['demand_total'] = demand_scn['demand_q'] * demand_scn['scaling_factor_demand']
    demand_scn = demand_scn.groupby(['abbreviation', 'IMPACT_code', 'kcal_scn', 'SSP_scn', 'diet_scn', 
                                     'year'])[['demand_total', 'demand_q']].sum().reset_index()
    demand_scn['scaling_factor_demand'] = demand_scn['demand_total'] / demand_scn['demand_q']
    demand_scn = demand_scn.drop(['demand_total', 'demand_q'], axis=1)
    return demand_scn

def aggregate_supply_scn(supply_scn, df_country, area_codes):
    supply_scn = supply_scn.merge(df_country[['abbreviation', 'yield_t_ha', 'supply_q']]).merge(area_codes).drop('abbreviation', axis=1).rename(
        columns={'abbreviation_new': 'abbreviation'})
    supply_scn['supply_total'] = supply_scn['supply_q'] * supply_scn['scaling_factor_supply']
    supply_scn['yield_new'] = supply_scn['yield_t_ha'] * supply_scn['scaling_factor_yield']
    supply_scn['area_2020'] = supply_scn['supply_q'] / supply_scn['yield_t_ha']
    supply_scn['area_new'] = supply_scn['supply_total'] / supply_scn['yield_new']
    supply_scn = supply_scn.groupby(['abbreviation', 'IMPACT_code', 'SSP_scn', 'RCP', 
                                     'year'])[['supply_total', 'supply_q', 'area_new', 'area_2020']].sum().reset_index()
    supply_scn['yield_new'] = supply_scn['supply_total'] / supply_scn['area_new']
    supply_scn['yield_2020'] = supply_scn['supply_q'] / supply_scn['area_2020']
    supply_scn['scaling_factor_supply'] = supply_scn['supply_total'] / supply_scn['supply_q']
    supply_scn['scaling_factor_yield'] = supply_scn['yield_new'] / supply_scn['yield_2020']
    supply_scn = supply_scn.drop(['supply_total', 'supply_q', 'area_new', 'area_2020', 'yield_new', 'yield_2020'], axis=1).fillna(1)
    return supply_scn


def get_imp_exp(category, area_codes):

    df = pd.read_csv(f'../../OPSIS/Data/FAOSTAT/FAO_re_export/supply_matrix_{category}_2018_2022.csv')
    df = df.rename(columns={'iso3': 'from_iso3'})
    df = df.melt(id_vars=['from_iso3'], value_vars=df['from_iso3'].values.tolist()).rename(
        columns={'variable': 'to_iso3', 'value': 'flow'})
    
    # aggregate flows by abbreviation, get imports and exports
    df = df.merge(area_codes, left_on='from_iso3', right_on='iso3').drop('iso3', axis=1).rename(
        columns={'abbreviation_new': 'from_abbreviation'}).merge(area_codes, left_on='to_iso3', right_on='iso3').drop(
        'iso3', axis=1).rename(columns={'abbreviation_new': 'to_abbreviation'})
    df = df.groupby(['from_abbreviation', 'to_abbreviation'])[['flow']].sum().reset_index()
    
    imports = df[df['from_abbreviation']!=df['to_abbreviation']].groupby(
        ['to_abbreviation'])[['flow']].sum().reset_index().rename(
        columns={'to_abbreviation': 'abbreviation', 'flow': 'import_q'})
    exports = df[df['from_abbreviation']!=df['to_abbreviation']].groupby(
        ['from_abbreviation'])[['flow']].sum().reset_index().rename(
        columns={'from_abbreviation': 'abbreviation', 'flow': 'export_q'})
    
    imp_exp = imports.merge(exports)
    
    return imp_exp

def aggregate_country_data(df_country, area_codes, imp_exp):
    # weigh producer price, yield by 2020 supply, demand and supply elas by 2020 demand and supply, recalculate import export domestic_q from flows
    df_country = df_country.merge(area_codes).drop('abbreviation', axis=1).rename(columns={'abbreviation_new': 'abbreviation'})

    # yield and prod price
    df_country['area'] = df_country['supply_q'] / df_country['yield_t_ha']
    df_country['supply_dup'] = df_country['supply_q']
    df_country['demand_dup'] = df_country['demand_q']
    df_country.loc[df_country['area']==0, 'area'] = 0.01 # to avoid getting nulls when aggregating 
    df_country.loc[df_country['supply_dup']==0, 'supply_dup'] = 0.01 # to avoid getting nulls when aggregating 
    df_country.loc[df_country['demand_dup']==0, 'demand_dup'] = 0.01 # to avoid getting nulls when aggregating 
    df_country['total_yield'] = df_country['yield_t_ha'] * df_country['area']
    df_country['total_price'] = df_country['prod_price_USD_t'] * df_country['supply_dup']

    # demand_elas and supply_elas
    df_country['demand_elas'] = df_country['demand_elas'] * df_country['demand_dup']
    df_country['supply_elas'] = df_country['supply_elas'] * df_country['supply_dup']

    df_country = df_country.groupby(['abbreviation', 'IMPACT_code'])[['total_yield', 'total_price', 'demand_q', 'supply_q',
                                                                      'demand_elas', 'supply_elas', 
                                                                      'area', 'demand_dup', 'supply_dup']].sum().reset_index()
    df_country['yield_t_ha'] = df_country['total_yield'] / df_country['area']
    df_country['prod_price_USD_t'] = df_country['total_price'] / df_country['supply_dup']
    df_country['demand_elas'] = df_country['demand_elas'] / df_country['demand_dup']
    df_country['supply_elas'] = df_country['supply_elas'] / df_country['supply_dup']

    df_country = df_country.drop(['total_yield', 'total_price', 'area', 'demand_dup', 'supply_dup'], axis=1).merge(imp_exp)
    df_country['domestic_q'] = df_country['supply_q'] - df_country['export_q']
    
    return df_country
 
if __name__ == '__main__':

    area_codes = pd.read_csv('../../OPSIS/Data/Country_group/regions_grouped.csv').rename(
        columns= {'Abbreviation': 'abbreviation'})[['abbreviation', 'iso3', 'abbreviation_new']]
    
    demand_elas = pd.read_csv(f'../../OPSIS/Data/Trade_clearance_model/Input/Future_scenarios/SSP2/IMPACT_future_demand_elas.csv')
    supply_elas = pd.read_csv(f'../../OPSIS/Data/Trade_clearance_model/Input/Future_scenarios/SSP2/IMPACT_supply_elas.csv')
    dem_sup_all = []

    for category in  ['jwhea', 'jrice', 'jmaiz', 'jbarl', 'jmill', 'jsorg', 'jocer', 'jcass', 
                      'jpota', 'jyams', 'jswpt', 'jorat', 'jvege', 'jbana', 'jplnt', 'jsubf', 
                      'jtemf', 'jbean', 'jchkp', 'jcowp', 'jlent', 'jpigp', 'jopul', 'jsoyb', 
                      'jgrnd', 'jothr', 'jrpsd', 'jsnfl', 'jtols', 'jpalm', 'jsugb', 'jsugc']:

        df_country = pd.read_csv(f'../../OPSIS/Data/Trade_clearance_model/Input/Country_data/country_information_{category}.csv')
        df_bilateral = pd.read_csv(f'../../OPSIS/Data/Trade_clearance_model/Input/Trade_cost/bilateral_trade_cost_{category}.csv')
        demand_scn = pd.read_csv(f'../../OPSIS/Data/Trade_clearance_model/Input/Future_scenarios/SSP2/demand_scn/IMPACT_future_demand_{category}.csv')
        supply_scn = pd.read_csv(f'../../OPSIS/Data/Trade_clearance_model/Input/Future_scenarios/SSP2/supply_scn/IMPACT_future_supply_{category}.csv')
        imp_exp = get_imp_exp(category, area_codes.drop('abbreviation', axis=1).drop_duplicates())

        dem_sup = df_country[['abbreviation', 'IMPACT_code', 'demand_q', 'supply_q']]
        dem_sup_all.append(dem_sup)

        df_bilateral = aggregate_bilateral_trade(df_bilateral, df_country, area_codes.drop('iso3', axis=1).drop_duplicates())
        demand_scn = aggregate_demand_scn(demand_scn, df_country, area_codes.drop('iso3', axis=1).drop_duplicates())
        supply_scn = aggregate_supply_scn(supply_scn, df_country, area_codes.drop('iso3', axis=1).drop_duplicates())
        df_country = aggregate_country_data(df_country, area_codes.drop('iso3', axis=1).drop_duplicates(), imp_exp)

        df_country.to_csv(f'../../OPSIS/Data/Trade_clearance_model/Grouped_Input/Country_data/country_information_{category}.csv', index=False)
        df_bilateral.to_csv(f'../../OPSIS/Data/Trade_clearance_model/Grouped_Input/Trade_cost/bilateral_trade_cost_{category}.csv', index=False)
        demand_scn.to_csv(f'../../OPSIS/Data/Trade_clearance_model/Grouped_Input/Future_scenarios/SSP2/demand_scn/IMPACT_future_demand_{category}.csv', index=False)
        supply_scn.to_csv(f'../../OPSIS/Data/Trade_clearance_model/Grouped_Input/Future_scenarios/SSP2/supply_scn/IMPACT_future_supply_{category}.csv', index=False)
        
    dem_sup_all = pd.concat(dem_sup_all, axis=0, ignore_index=True)

    demand_elas, supply_elas = aggregate_elas(demand_elas, supply_elas, dem_sup_all, area_codes.drop('iso3', axis=1).drop_duplicates())
    demand_elas.to_csv(f'../../OPSIS/Data/Trade_clearance_model/Grouped_Input/Future_scenarios/SSP2/IMPACT_future_demand_elas.csv', index=False)
    supply_elas.to_csv(f'../../OPSIS/Data/Trade_clearance_model/Grouped_Input/Future_scenarios/SSP2/IMPACT_supply_elas.csv', index=False)
    
        


