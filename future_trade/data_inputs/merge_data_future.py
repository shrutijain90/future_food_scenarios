# Usage: python -m future_trade.data_inputs.merge_data_future

import geopandas as gpd
import pandas as pd
import numpy as np
import os

from future_trade.data_inputs.merge_data import get_demand_elas, get_supply_elas
from future_trade.data_inputs.balance_trade import get_area_codes

def get_demand_scn(food_group, categories):
    
    cons = pd.read_csv('../../OPSIS/Data/Future_production_demand_data/demand_scn_food_feed_other.csv') # lancet scenarios with feed and other included
    cons = cons[(cons['food_group']==food_group) & (cons['year']>=2020)]
    cons['demand_q'] = cons['food_lancet_total_est'] + cons['feed_lancet_total_est'] + cons['other_fao_total'] # 'other_lancet_total_est'
    cons = cons[['region', 'kcal_scn', 'SSP_scn', 'diet_scn', 'year',
                 'food_group', 'demand_q']].rename(columns={'region': 'Abbreviation'})
    cons.loc[cons['demand_q']==0, 'demand_q'] = 1 #to prevent nulls and infs
    cons = cons.merge(cons[cons['year']==2020].rename(columns={'demand_q': 'demand_2020'}).drop('year', axis=1))
    cons['scaling_factor_demand'] = cons['demand_q'] / cons['demand_2020']
    cons = cons.drop(['demand_q', 'demand_2020', 'food_group'], axis=1)

    # to prevent the demand for some commodities (nuts_seeds and oil_veg) in some countries (LSO, TKM; BTN, COD, PNG, SOM) from exploding
    cons = cons.merge(cons[cons['scaling_factor_demand']<2000].groupby(['kcal_scn', 'diet_scn', 'year'])[[
        'scaling_factor_demand']].max().reset_index().rename(columns={'scaling_factor_demand': 'scaling_factor_demand_max'}))
    cons.head()
    cons.loc[cons['scaling_factor_demand']>2000, 'scaling_factor_demand'] = cons.loc[cons['scaling_factor_demand']>2000]['scaling_factor_demand_max']
    cons = cons.drop('scaling_factor_demand_max', axis=1)
    
    for category in categories:
        cons['IMPACT_code'] = category
        cons.rename(columns={
        'Abbreviation': 'abbreviation'
    }).to_csv(f'../../OPSIS/Data/Trade_clearance_model/Input/Future_scenarios/SSP2/demand_scn/IMPACT_future_demand_{category}.csv', index=False)

    return None

def get_impact_supply_data(FAO_area_codes):
    crop_dict = {
        'CER-Barley' : 'jbarl', 
        'CER-Maize' : 'jmaiz', 
        'CER-Millet' : 'jmill', 
        'CER-Rice' : 'jrice', 
        'CER-Wheat': 'jwhea', 
        'CER-Sorghum' : 'jsorg',
        'CER-Other Cereals' : 'jocer',

        'RTB-Potato': 'jpota',
        'RTB-Cassava': 'jcass',  
        'RTB-Sweet Potato': 'jswpt',
        'RTB-Yams': 'jyams',
        'RTB-Other Roots' : 'jorat',

        'F&V-Temperate Fruit' : 'jtemf', 
        'F&V-Tropical Fruit' : 'jsubf',
        'F&V-Vegetables' : 'jvege', 
        'RTB-Banana' : 'jbana',
        'RTB-Plantain' : 'jplnt',

        'PUL-Other ': 'jopul', 
        'PUL-Beans': 'jbean', 
        'PUL-Chickpeas': 'jchkp', 
        'PUL-Lentils' : 'jlent',
        'PUL-Pigeonpeas' : 'jpigp',
        'PUL-Cowpeas' : 'jcowp', 

        'O&S-Groundnut' : 'jgrnd',
        'OTH-Other': 'jothr',

        'O&S-Soybean' : 'jsoyb',
        'O&S-Other Oilseeds': 'jtols', 
        'O&S-Rapeseed' : 'jrpsd',
        'O&S-Sunflower' : 'jsnfl',

        'O&S-Sugar beet': 'jsugb',
        'O&S-Sugarcane': 'jsugc',
        'O&S-Palm Fruit': 'jpalm'
        }

    var_dict = {
        'Harvested area  (000 Ha)': 'Area',
        'Yield (t/ha)': 'Yield',
        'Production (000 ton)': 'Production'
    }
    
    impact_supply = pd.read_csv('../../OPSIS/Data/Future_production_demand_data/impact_results.csv')
    impact_supply['description'] = impact_supply['description'].map(var_dict)
    impact_supply['name'] = impact_supply['name'].map(crop_dict)
    impact_supply = impact_supply[(impact_supply['name'].notnull()) 
                                  & (impact_supply['type']=='Overall') 
                                  & (impact_supply['region'].isin(FAO_area_codes['Abbreviation'].unique()))]
    impact_supply = impact_supply.pivot(index=['region', 'name', 'yrs', 'ssp', 'gcm', 'rcp'], 
                                        columns='description', values='value').reset_index()
    impact_supply = impact_supply.rename(columns={'region': 'abbreviation', 'name': 'IMPACT_code', 'yrs': 'year',
                                                  'ssp': 'SSP_scn', 'rcp': 'RCP'})
    return impact_supply

def get_supply_scn(impact_supply, category, FAO_area_codes):

    df_2020 = impact_supply[impact_supply['year']==2020]
    df_future = impact_supply[impact_supply['year'].isin([2020, 2025, 2030, 2035, 2040, 2045, 2050])]
    df_2020 = df_2020.groupby(['abbreviation', 'IMPACT_code'])[['Production', 'Yield']].mean().reset_index()
    df_future = df_future.groupby(['abbreviation', 'IMPACT_code', 'year', 'SSP_scn', 'RCP'])[['Production', 'Yield']].mean().reset_index()
    df_future = df_future.merge(df_2020.rename(columns={
        'Yield': 'Yield_2020',
        'Production': 'Production_2020'
    }))
    
    df_future = df_future[df_future['IMPACT_code']==category]
    df_future['scaling_factor_yield'] = df_future['Yield'] / df_future['Yield_2020']
    df_future['scaling_factor_supply'] = df_future['Production'] / df_future['Production_2020']
    df_future = df_future.drop(['Yield', 'Yield_2020', 'Production', 'Production_2020'], axis=1)
    
    def _add_regions(g):
        g = g.merge(FAO_area_codes.rename(columns={'Abbreviation': 'abbreviation'})[['abbreviation']].drop_duplicates(), how='right')
        g['IMPACT_code'] = g[g['IMPACT_code'].notnull()]['IMPACT_code'].values[0]
        g['year'] = g[g['year'].notnull()]['year'].values[0]
        g['SSP_scn'] = g[g['SSP_scn'].notnull()]['SSP_scn'].values[0]
        g['RCP'] = g[g['RCP'].notnull()]['RCP'].values[0]
        g = g.fillna(1).reset_index(drop=True)
        return g

    df_future = df_future.groupby(['IMPACT_code', 'year', 'SSP_scn', 'RCP']).apply(lambda g: _add_regions(g)).reset_index(drop=True)
    df_future.loc[df_future['year']==2020, 'scaling_factor_yield'] = 1
    df_future.loc[df_future['year']==2020, 'scaling_factor_supply'] = 1
    df_future.to_csv(f'../../OPSIS/Data/Trade_clearance_model/Input/Future_scenarios/SSP2/supply_scn/IMPACT_future_supply_{category}.csv', index=False)

    return None

if __name__ == '__main__':

    items_dict = {
        'wheat': ['jwhea'],
        'rice' : ['jrice'],
        'maize' : ['jmaiz'],
        'othr_grains' : ['jbarl', 'jmill', 'jsorg', 'jocer'],
        'roots' : ['jcass', 'jpota', 'jswpt', 'jyams', 'jorat'],
        'vegetables' : ['jvege'],
        'fruits' : ['jbana', 'jplnt', 'jsubf', 'jtemf'],
        'legumes' : ['jbean', 'jchkp', 'jcowp', 'jlent', 'jpigp', 'jopul'],
        'soybeans' : ['jsoyb'],
        'nuts_seeds' : ['jgrnd', 'jothr'],
        'oil_veg' : ['jrpsd', 'jsnfl', 'jtols'],
        'oil_palm' : ['jpalm'],
        'sugar' : ['jsugb', 'jsugc']
    }
    
    FAO_area_codes = get_area_codes()
    demand_elas = get_demand_elas()
    supply_elas = get_supply_elas()
    impact_supply = get_impact_supply_data(FAO_area_codes)
    
    # exporting future demand elasticities
    demand_elas['scaling_factor_demand_elas'] = demand_elas['demand_elas'] / demand_elas['demand_elas_baseline']
    demand_elas.rename(columns={
        'Abbreviation': 'abbreviation'
    }).to_csv('../../OPSIS/Data/Trade_clearance_model/Input/Future_scenarios/SSP2/IMPACT_future_demand_elas.csv', index=False)
    # exporting supply elasticities
    supply_elas.rename(columns={
        'Abbreviation': 'abbreviation'
    }).to_csv('../../OPSIS/Data/Trade_clearance_model/Input/Future_scenarios/SSP2/IMPACT_supply_elas.csv', index=False)

    for food_group in items_dict.keys():
        # exporting demand scenarios 
        print(food_group)
        _ = get_demand_scn(food_group, items_dict[food_group])
        for category in items_dict[food_group]:
            # exporting supply scenarios
            _ = get_supply_scn(impact_supply, category, FAO_area_codes)
            
    