# Usage: python -m future_trade.data_inputs.merge_data

import geopandas as gpd
import pandas as pd
import numpy as np
import os

def get_area_codes():

    FAO_area_codes = pd.read_csv('../../OPSIS/Data/Country_group/regions.csv')
    FAO_area_codes = FAO_area_codes[['Abbreviation', 'M49 Code', 'iso3']]
    # removing countries which don't have corresponding FBS/SUA or consumption data - leaves a total of 153 unique regions (Abbreviation is the unique identifier here)
    FAO_area_codes = FAO_area_codes[~FAO_area_codes['Abbreviation'].isin(['ERI', 'GNQ', 'OSA', 'PSE', 'SSD'])] 
    FAO_area_codes = FAO_area_codes.sort_values(by='Abbreviation').reset_index(drop=True)
    
    return FAO_area_codes

def _melt_matrix(df, var, trade_thresh):
    
    df = df.rename(columns={'Abbreviation': 'from_abbreviation'})
    df = df.melt(id_vars=['from_abbreviation'], value_vars=df['from_abbreviation'].values.tolist()).rename(
        columns={'variable': 'to_abbreviation', 'value': var})
    df = df[df['from_abbreviation']!=df['to_abbreviation']]  
    df.loc[df[var]<trade_thresh, var] = 0    
    
    return df

# flows data
def get_flows(category, trade_thresh=10):

    flows_2017_2021 = pd.read_csv(f'../../OPSIS/Data/FAOSTAT/FAO_re_export/supply_matrix_{category}_2017_2021.csv')
    flows_2017_2021 = _melt_matrix(flows_2017_2021, 'flow_2017_2021', trade_thresh)
    
    flows_2012_2016 = pd.read_csv(f'../../OPSIS/Data/FAOSTAT/FAO_re_export/supply_matrix_{category}_2012_2016.csv')
    flows_2012_2016 = _melt_matrix(flows_2012_2016, 'flow_2012_2016', trade_thresh)
    
    flows = flows_2012_2016.merge(flows_2017_2021)
    flows['item'] = category
    
    imports = flows.groupby(['to_abbreviation', 'item'])[['flow_2017_2021']].sum().reset_index().rename(
        columns={'to_abbreviation': 'Abbreviation', 'flow_2017_2021': 'imports'})
    exports = flows.groupby(['from_abbreviation', 'item'])[['flow_2017_2021']].sum().reset_index().rename(
        columns={'from_abbreviation': 'Abbreviation', 'flow_2017_2021': 'exports'})
    imp_exp = imports.merge(exports)
    
    return flows, imp_exp

# trade data
def get_trade(category, trade_thresh=10):

    trade_2017_2021 = pd.read_csv(f'../../OPSIS/Data/FAOSTAT/FAO_bal_trade_mat/trade_matrix_{category}_2017_2021.csv')
    trade_2017_2021 = _melt_matrix(trade_2017_2021, 'trade_2017_2021', trade_thresh)
    
    trade_2012_2016 = pd.read_csv(f'../../OPSIS/Data/FAOSTAT/FAO_bal_trade_mat/trade_matrix_{category}_2012_2016.csv')
    trade_2012_2016 = _melt_matrix(trade_2012_2016, 'trade_2012_2016', trade_thresh)
    
    trade = trade_2012_2016.merge(trade_2017_2021)
    trade['item'] = category
    
    return trade


def get_prod(category):

    prod = pd.read_csv(f'../../OPSIS/Data/FAOSTAT/FAO_prod_mat/prod_matrix_{category}_2017_2021.csv')
    prod['item'] = category

    return prod

def get_prod_prices(category):

    prod_prices = pd.read_csv(f'../../OPSIS/Data/FAOSTAT/FAO_prod_prices/prod_prices_{category}.csv')
    prod_prices['item'] = category

    return prod_prices

def get_transport_data(FAO_area_codes):
    
    transport_admin = pd.read_parquet('../../data/transport_data/global_lowest_transport_cost.parquet')
    transport_admin = transport_admin.merge(FAO_area_codes, left_on='from_iso3', right_on='iso3').drop(
        ['iso3', 'M49 Code'], axis=1).rename(columns={'Abbreviation': 'from_abbreviation'})
    transport_admin = transport_admin.merge(FAO_area_codes, left_on='to_iso3', right_on='iso3').drop(
        ['iso3', 'M49 Code'], axis=1).rename(columns={'Abbreviation': 'to_abbreviation'})
    transport_admin = transport_admin[transport_admin['from_abbreviation']!=transport_admin['to_abbreviation']]
    transport_admin.loc[transport_admin['customs_cost'].isna(), 'customs_cost'] = 0
    transport = transport_admin.groupby(['from_abbreviation', 'to_abbreviation'])[[
        'freight_USD_t', 'transport_USD_t', 'time_h', 'distance_km','border_USD_t', 'trade_USD_t', 'customs_cost']].mean().reset_index()
    
    return transport

def get_tariffs_data(transport, FAO_area_codes, mm, codes):

    # import tariffs data - codes based on https://comtradeplus.un.org/ListOfReferences 
    mm = mm[(mm['hs6_rev2017'].isin(codes))]
    mm_country = pd.read_csv('../../data/Import_tariffs/Countries.csv')
    mm_country = mm_country.merge(FAO_area_codes[['iso3', 'Abbreviation']])
    
    mm = mm.merge(mm_country[['code', 'Abbreviation']], right_on='code', left_on='importer').drop(
        'code', axis=1).rename(columns={'Abbreviation': 'to_abbreviation'})
    mm = mm.merge(mm_country[['code', 'Abbreviation']], right_on='code', left_on='exporter').drop(
        'code', axis=1).rename(columns={'Abbreviation': 'from_abbreviation'})
    mm = mm.drop(['importer', 'exporter', 'hs6_rev2017'], axis=1)
    mm = mm.groupby(['from_abbreviation','to_abbreviation']).mean()[['Pref_Applied_AVE']].reset_index()

    # combining transport and tariffs data 
    transport = transport.merge(mm, how='left')
    transport['import_tariff'] = transport['Pref_Applied_AVE'] * 0.8 * (transport['transport_USD_t'] + transport['border_USD_t'])
    
    return transport

if __name__ == '__main__':
    
    items_dict = {# wheat
                  'jwhea': [100111, 100119, 100191, 100199],
                  # rice
                  'jrice': [100610, 100620, 100630, 100640],
                  # maize
                  'jmaiz': [100510, 100590],
                  # othr_grains
                  'jbarl': [100310, 100390],
                  'jmill': [100821, 100829],
                  'jsorg': [100710, 100790],
                  'jocer': [100210, 100290, 100410, 100490, 100810, 
                            100830, 100840, 100850, 100860, 100890], 
                  # roots
                  'jcass': [71410],
                  'jpota': [70110, 70190, 71010],
                  'jyams': [71430],
                  'jswpt': [71420],
                  'jorat': [71440, 71450, 71490],
                  # vegetables
                  'jvege': [70200, 70310, 70320, 70390, 70410, 70420, 
                            70490, 70511, 70519, 70521, 70529, 70610, 
                            70690, 70700, 70810, 70820, 70890, 70920, 
                            70930, 70940, 70951, 70959, 70960, 70970, 
                            70991, 70992, 70993, 70999, 71021, 71029, 
                            71030, 71040, 71080, 71140, 71151, 71159, 
                            71190, 71220, 71231, 71232, 71233, 71239, 
                            71290], 
                  # fruits
                  'jbana': [80390],
                  'jplnt': [80310],
                  'jsubf': [80410, 80420, 80430, 80440, 80450, 80510, 
                            80521, 80522, 80529, 80540, 80550, 80590, 
                            80711, 80719, 80720, 80910, 81050, 81060, 
                            81090, 81310, 80111, 80112, 80119],
                  'jtemf': [71120, 80610, 80620, 80810, 80830, 80840,  
                            80921, 80929, 80930, 80940, 81010, 81020, 
                            81030, 81040, 81070, 81110, 81120, 81190, 
                            81210, 81290, 81320, 81330, 81340], 
                  # legumes
                  'jbean': [71331, 71332, 71333, 71334, 71350],
                  'jchkp': [71320],
                  'jcowp': [71335],
                  'jlent': [71340],
                  'jpigp': [71360],
                  'jopul': [71310, 71339, 71390, 110610],
                  # soybeans
                  'jsoyb': [120110, 120190, 120810, 150710, 150790, 210310, 230400], 
                  # nuts_seeds
                  'jgrnd': [120230, 120241, 120242],
                  'jothr': [81350, 80121, 80122, 80131, 80132, 80211,
                            80212, 80231, 80232, 80241, 80242, 80251, 
                            80252, 80261, 80262, 80270, 80280, 80290, 
                            120400, 120600, 120740, 120760, 120791, 120799], 
                  # oil_veg
                  'jrpsd': [100510, 120590, 151411, 151419, 151491, 151499],
                  'jsnfl': [120600, 151211, 151219],
                  'jtols': [120230, 120241, 120242, 120400, 120721, 
                            120729, 120730, 120740, 120750, 120760,
                            120799, 150810, 150890, 150910, 150990, 
                            151000, 151221, 151229, 151311, 151319, 
                            151511, 151519, 151530, 151550, 151590],
                  # oil_palm
                  'jpalm': [120710, 151110, 151190, 151321, 151329],
                  # sugar
                  'jsugb': [121291, 170112],
                  'jsugc': [121293, 170113, 170114] 
    }
   
    FAO_area_codes = get_area_codes()
    transport = get_transport_data(FAO_area_codes)
    mm_2019 = pd.read_csv('../../data/Import_tariffs/mm2019.csv', sep=';')
    
    
    for category in items_dict.keys():
        print(category)
        
        flows, imp_exp = get_flows(category, trade_thresh=1)
        trade = get_trade(category, trade_thresh=1)
        prod_prices = get_prod_prices(category) # in USD per ha
        tariffs = get_tariffs_data(transport, FAO_area_codes, mm_2019, items_dict[category])

        df_country = prod_prices.merge(imp_exp)
        df_country['Consumption'] = df_country['Production'] + df_country['imports'] - df_country['exports'] 

        df_bilateral = flows.merge(trade).merge(tariffs)
        
        df_country[['Abbreviation', 'item', 'Area', 'Production', 'imports', 'exports', 'Consumption', 
                    'Producer_price']].to_parquet(f'../../OPSIS/Data/FAOSTAT/combined_data/combined_inputs_country_{category}.parquet.gzip', index=False, compression='gzip')
        
        df_bilateral[['from_abbreviation', 'to_abbreviation', 'item', 
                      'flow_2012_2016', 'flow_2017_2021', 'trade_2012_2016', 'trade_2017_2021',
                      'freight_USD_t', 'transport_USD_t', 'time_h', 'distance_km',
                      'border_USD_t', 'trade_USD_t', 'customs_cost', 'Pref_Applied_AVE',
                      'import_tariff']].to_parquet(f'../../OPSIS/Data/FAOSTAT/combined_data/combined_inputs_bilateral_{category}.parquet.gzip', index=False, compression='gzip')
        