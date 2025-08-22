# Usage: python -m future_trade.data_inputs.merge_data_future

import geopandas as gpd
import pandas as pd
import numpy as np
import os

from future_trade.data_inputs.merge_data import get_demand_elas, get_supply_elas

def get_demand_scn(food_group, categories):
    
    cons = pd.read_csv('../../OPSIS/Data/Future_production_demand_data/demand_scn_food_feed_other.csv') # lancet scenarios with feed and other included
    cons = cons[(cons['food_group']==food_group) & (cons['year']>=2020)]
    cons['demand_q'] = cons['food_lancet_total_est'] + cons['feed_lancet_total_est'] + cons['other_fao_total'] # 'other_lancet_total_est'
    cons = cons[['region', 'kcal_scn', 'SSP_scn', 'diet_scn', 'year',
                 'food_group', 'demand_q']].rename(columns={'region': 'Abbreviation'})
    cons = cons.merge(cons[cons['year']==2020].rename(columns={'demand_q': 'demand_2020'}).drop('year', axis=1))
    cons['scaling_factor_demand'] = cons['demand_q'] / cons['demand_2020']
    cons = cons.drop(['demand_q', 'demand_2020', 'food_group'], axis=1)
    
    for category in categories:
        cons['IMPACT_code'] = category
        cons.rename(columns={
        'Abbreviation': 'abbreviation'
    }).to_csv(f'../../OPSIS/Data/Trade_clearance_model/Input/Future_scenarios/SSP2/demand_scn/IMPACT_future_demand_{category}.csv', index=False)

    return None

def get_impact_supply_data():

    crop_codes = ['jwhea', 'jrice', 'jmaiz', 'jbarl', 'jmill', 'jsorg', 'jocer', 'jcass', 
                  'jpota', 'jyams', 'jswpt', 'jorat', 'jvege', 'jbana', 'jplnt', 'jsubf', 
                  'jtemf', 'jbean', 'jchkp', 'jcowp', 'jlent', 'jpigp', 'jopul', 'jsoyb', 
                  'jgrnd', 'jothr', 'jrpsd', 'jsnfl', 'jtols', 'jpalm', 'jsugb', 'jsugc']

    basins = pd.read_excel('../../OPSIS/Data/Future_production_demand_data/IMPACT-master/DriverAssumptions/CorrespondenceFiles/Sets.xlsx',
                           sheet_name='Regions', header=3)
    basins = basins[['FPU.1', 'CTY.2']].rename(columns={'FPU.1': 'FPU', 'CTY.2': 'Abbreviation'})

    # baseline
    baseline = pd.read_csv('../../OPSIS/Data/Future_production_demand_data/IMPACT-master/DriverAssumptions/BaseYearData/BaseFPU_Crops.csv', 
                              header=None)
    baseline.columns = ['IMPACT_code','FPU','Irr','variable', 'value']
    baseline = baseline[baseline['IMPACT_code'].isin(crop_codes)]
    baseline = baseline.merge(basins)
    baseline = baseline[~baseline['Abbreviation'].isin(['ERI', 'GNQ', 'OSA', 'PSE', 'GRL'])]
    baseline['Irr'] = 'a' + baseline['Irr']
    baseline = baseline.pivot(index=['IMPACT_code', 'FPU', 'Irr', 'Abbreviation'], columns='variable', values='value').reset_index()
    baseline = baseline.rename(columns={'ARA': 'area', 'QS': 'production', 'YLD': 'yield'})
    baseline['area'] = baseline['area'] * 1e3 #converting to ha
    baseline['production'] = baseline['production'] * 1e3 #converting to tonnes 

    # yield growth
    yield_gr = pd.read_csv('../../OPSIS/Data/Future_production_demand_data/IMPACT-master/DriverAssumptions/ProductionGrowth_Yield/YLDGR.csv', 
                              header=None)
    yield_gr.columns = ['IMPACT_code','FPU','Irr','period', 'yield_gr']
    yield_gr = yield_gr[yield_gr['IMPACT_code'].isin(crop_codes)]
    yield_gr = yield_gr.merge(basins)
    yield_gr = yield_gr[~yield_gr['Abbreviation'].isin(['ERI', 'GNQ', 'OSA', 'PSE', 'GRL'])]

    # area growth
    area_gr = pd.read_csv('../../OPSIS/Data/Future_production_demand_data/IMPACT-master/DriverAssumptions/ProductionGrowth_Area/AREAGR.csv', 
                              header=None)
    area_gr.columns = ['IMPACT_code','FPU','Irr','period', 'area_gr']
    area_gr = area_gr[area_gr['IMPACT_code'].isin(crop_codes)]
    area_gr = area_gr.merge(basins)
    area_gr = area_gr[~area_gr['Abbreviation'].isin(['ERI', 'GNQ', 'OSA', 'PSE', 'GRL'])]

    # climate impacts
    cc = pd.read_csv('../../OPSIS/Data/Future_production_demand_data/IMPACT-master/DriverAssumptions/ClimateImpacts_Yield/CCDelta.csv',
                     header=None)
    cc.columns = ['GCM','RCP', 'model', 'IMPACT_code', 'FPU', 'Irr', 'cc']
    cc = cc[cc['IMPACT_code'].isin(crop_codes)]
    cc = cc.merge(basins)
    cc = cc[~cc['Abbreviation'].isin(['ERI', 'GNQ', 'OSA', 'PSE', 'GRL'])]
    ##### CONSIDERING CO2 OFF. CAN CHANGE HERE
    cc = cc[cc['model']=='dssat_co2_379']
    cc = cc.groupby(['Abbreviation', 'IMPACT_code', 'RCP', 'FPU', 'Irr'])[['cc']].mean().reset_index()

    return baseline, yield_gr, area_gr, cc


def get_supply_scn(baseline, yield_gr, area_gr, cc, category):

    # remaining questions
    # 1. units of yield growth, area growth, climate impacts
    # 2. growth rates provided when there is no baseline?
    # 3. co2 on or off for climate impacts
    # 4. climate impacts on years prior to 2050?
    # 5. Are the growth rates and climate impacts applicable to all SSPs? We're using SSP2. 
    
    # first, no climate change, only tech growth

    basins = pd.read_excel('../../OPSIS/Data/Future_production_demand_data/IMPACT-master/DriverAssumptions/CorrespondenceFiles/Sets.xlsx',
                           sheet_name='Regions', header=3)
    basins = basins[['FPU.1', 'CTY.2']].rename(columns={'FPU.1': 'FPU', 'CTY.2': 'Abbreviation'})
    df = baseline[baseline['IMPACT_code']==category].reset_index(drop=True)
    df = df.merge(basins, how='outer')
    df = df[df['Abbreviation'].isin(baseline['Abbreviation'].unique())]
    df['IMPACT_code']=category
    df.loc[df['Irr'].isna(), 'Irr'] = 'arf'

    df['year'] = 2005
    
    df_10 = df.merge(area_gr[(area_gr['period']=='2005-2010') & (area_gr['IMPACT_code']==category)], how='outer').merge(
        yield_gr[(yield_gr['period']=='2005-2010') & (yield_gr['IMPACT_code']==category)], how='outer').fillna(0)
    df_10['year'] = 2010
    df_10['area'] = df_10['area'] * (1 + 0.01 * df_10['area_gr'])
    df_10['yield'] = df_10['yield'] * (1 + 0.01 * df_10['yield_gr'])
    df_10['production'] = df_10['yield'] * df_10['area']
    df_10 = df_10.drop(['area_gr', 'yield_gr', 'period'], axis=1)
    
    df_15 = df_10.merge(area_gr[(area_gr['period']=='2010-2015') & (area_gr['IMPACT_code']==category)], how='outer').merge(
        yield_gr[(yield_gr['period']=='2010-2015') & (yield_gr['IMPACT_code']==category)], how='outer').fillna(0)
    df_15['year'] = 2015
    df_15['area'] = df_15['area'] * (1 + 0.01 * df_15['area_gr'])
    df_15['yield'] = df_15['yield'] * (1 + 0.01 * df_15['yield_gr'])
    df_15['production'] = df_15['yield'] * df_15['area']
    df_15 = df_15.drop(['area_gr', 'yield_gr', 'period'], axis=1)
    
    df_20 = df_15.merge(area_gr[(area_gr['period']=='2015-2020') & (area_gr['IMPACT_code']==category)], how='outer').merge(
        yield_gr[(yield_gr['period']=='2015-2020') & (yield_gr['IMPACT_code']==category)], how='outer').fillna(0)
    df_20['year'] = 2020
    df_20['area'] = df_20['area'] * (1 + 0.01 * df_20['area_gr'])
    df_20['yield'] = df_20['yield'] * (1 + 0.01 * df_20['yield_gr'])
    df_20['production'] = df_20['yield'] * df_20['area']
    df_20 = df_20.drop(['area_gr', 'yield_gr', 'period'], axis=1)
    
    df_25 = df_20.merge(area_gr[(area_gr['period']=='2020-2025') & (area_gr['IMPACT_code']==category)], how='outer').merge(
        yield_gr[(yield_gr['period']=='2020-2025') & (yield_gr['IMPACT_code']==category)], how='outer').fillna(0)
    df_25['year'] = 2025
    df_25['area'] = df_25['area'] * (1 + 0.01 * df_25['area_gr'])
    df_25['yield'] = df_25['yield'] * (1 + 0.01 * df_25['yield_gr'])
    df_25['production'] = df_25['yield'] * df_25['area']
    df_25 = df_25.drop(['area_gr', 'yield_gr', 'period'], axis=1)
    
    df_30 = df_25.merge(area_gr[(area_gr['period']=='2025-2030') & (area_gr['IMPACT_code']==category)], how='outer').merge(
        yield_gr[(yield_gr['period']=='2025-2030') & (yield_gr['IMPACT_code']==category)], how='outer').fillna(0)
    df_30['year'] = 2030
    df_30['area'] = df_30['area'] * (1 + 0.01 * df_30['area_gr'])
    df_30['yield'] = df_30['yield'] * (1 + 0.01 * df_30['yield_gr'])
    df_30['production'] = df_30['yield'] * df_30['area']
    df_30 = df_30.drop(['area_gr', 'yield_gr', 'period'], axis=1)
    
    df_35 = df_30.merge(area_gr[(area_gr['period']=='2030-2035') & (area_gr['IMPACT_code']==category)], how='outer').merge(
        yield_gr[(yield_gr['period']=='2030-2035') & (yield_gr['IMPACT_code']==category)], how='outer').fillna(0)
    df_35['year'] = 2035
    df_35['area'] = df_35['area'] * (1 + 0.01 * df_35['area_gr'])
    df_35['yield'] = df_35['yield'] * (1 + 0.01 * df_35['yield_gr'])
    df_35['production'] = df_35['yield'] * df_35['area']
    df_35 = df_35.drop(['area_gr', 'yield_gr', 'period'], axis=1)
    
    df_40 = df_35.merge(area_gr[(area_gr['period']=='2035-2040') & (area_gr['IMPACT_code']==category)], how='outer').merge(
        yield_gr[(yield_gr['period']=='2035-2040') & (yield_gr['IMPACT_code']==category)], how='outer').fillna(0)
    df_40['year'] = 2040
    df_40['area'] = df_40['area'] * (1 + 0.01 * df_40['area_gr'])
    df_40['yield'] = df_40['yield'] * (1 + 0.01 * df_40['yield_gr'])
    df_40['production'] = df_40['yield'] * df_40['area']
    df_40 = df_40.drop(['area_gr', 'yield_gr', 'period'], axis=1)
    
    df_45 = df_40.merge(area_gr[(area_gr['period']=='2040-2045') & (area_gr['IMPACT_code']==category)], how='outer').merge(
        yield_gr[(yield_gr['period']=='2040-2045') & (yield_gr['IMPACT_code']==category)], how='outer').fillna(0)
    df_45['year'] = 2045
    df_45['area'] = df_45['area'] * (1 + 0.01 * df_45['area_gr'])
    df_45['yield'] = df_45['yield'] * (1 + 0.01 * df_45['yield_gr'])
    df_45['production'] = df_45['yield'] * df_45['area']
    df_45 = df_45.drop(['area_gr', 'yield_gr', 'period'], axis=1)
    
    df_50 = df_45.merge(area_gr[(area_gr['period']=='2045-2050') & (area_gr['IMPACT_code']==category)], how='outer').merge(
        yield_gr[(yield_gr['period']=='2045-2050') & (yield_gr['IMPACT_code']==category)], how='outer').fillna(0)
    df_50['year'] = 2050
    df_50['area'] = df_50['area'] * (1 + 0.01 * df_50['area_gr'])
    df_50['yield'] = df_50['yield'] * (1 + 0.01 * df_50['yield_gr'])
    df_50['production'] = df_50['yield'] * df_50['area']
    df_50 = df_50.drop(['area_gr', 'yield_gr', 'period'], axis=1)

    df = pd.concat([df, df_10, df_15, df_20, df_25, df_30, df_35, df_40, df_45, df_50], axis=0, ignore_index=True)
    df = df[df['year']>=2020]
    
    # adding in climate change, assuming the same rate of change applies to all years in the future, 
    # ignoring impacts if no yields available in the first place
    df_cc1 = df.merge(cc[cc['RCP']=='rcp2p6'], how='left')
    df_cc1['RCP'] = 'rcp2p6'
    df_cc2 = df.merge(cc[cc['RCP']=='rcp4p5'], how='left')
    df_cc2['RCP'] = 'rcp4p5'
    df_cc3 = df.merge(cc[cc['RCP']=='rcp6p0'], how='left')
    df_cc3['RCP'] = 'rcp6p0'
    df_cc4 = df.merge(cc[cc['RCP']=='rcp8p5'], how='left')
    df_cc4['RCP'] = 'rcp8p5'
    df_cc = pd.concat([df_cc1, df_cc2, df_cc3, df_cc4], axis=0, ignore_index=True)

    df_cc = df_cc.fillna(0)
    df_cc['cc'] = 1 + df_cc['cc']
    df_cc.loc[df_cc['year']==2020, 'cc'] = 1 #no multiplication factor in the present
    df_cc['yield'] = df_cc['yield'] * df_cc['cc']
    df_cc['production'] = df_cc['yield'] * df_cc['area']
    df_cc = df_cc.drop('cc', axis=1)

    df['RCP'] = 'none'
    df = pd.concat([df, df_cc], axis=0, ignore_index=True)
    df['SSP_scn'] = 'SSP2' 

    # aggregating by country, climate_scn, and year
    df = df.groupby(['Abbreviation', 'IMPACT_code', 'year', 'RCP', 'SSP_scn'])[['area', 'production']].sum().reset_index()
    df['yield'] = df['production'] / df['area']
    df = df.fillna(0)

    # converting to scaling factors for yield and production
    df = df.drop('area', axis=1).merge(df[df['year']==2020].rename(
        columns={'production': 'production_2020', 'yield': 'yield_2020'}).drop(['area', 'year'], axis=1))
    df['scaling_factor_yield'] = df['yield'] / df['yield_2020']
    df['scaling_factor_supply'] = df['production'] / df['production_2020']
    df = df.drop(['yield', 'yield_2020', 'production', 'production_2020'], axis=1)
    df = df.fillna(1)
    df.rename(columns={
        'Abbreviation': 'abbreviation'
    }).to_csv(f'../../OPSIS/Data/Trade_clearance_model/Input/Future_scenarios/SSP2/supply_scn/IMPACT_future_supply_{category}.csv', index=False)

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
    
    demand_elas = get_demand_elas()
    supply_elas = get_supply_elas()
    baseline, yield_gr, area_gr, cc = get_impact_supply_data()
    
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
            _ = get_supply_scn(baseline, yield_gr, area_gr, cc, category)
            
    