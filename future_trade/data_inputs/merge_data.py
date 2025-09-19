# Usage: python -m future_trade.data_inputs.merge_data

import pandas as pd
import numpy as np
import os

from future_trade.data_inputs.balance_trade import get_area_codes

def _melt_matrix(df, var):
    
    df = df.rename(columns={'iso3': 'from_iso3'})
    df = df.melt(id_vars=['from_iso3'], value_vars=df['from_iso3'].values.tolist()).rename(
        columns={'variable': 'to_iso3', 'value': var})
    
    return df

# flows data
def get_flows(category):

    flows_2018_2022 = pd.read_csv(f'../../OPSIS/Data/FAOSTAT/FAO_re_export/supply_matrix_{category}_2018_2022.csv')
    flows_2018_2022 = _melt_matrix(flows_2018_2022, 'flow_2018_2022')

    flows_2013_2017 = pd.read_csv(f'../../OPSIS/Data/FAOSTAT/FAO_re_export/supply_matrix_{category}_2013_2017.csv')
    flows_2013_2017 = _melt_matrix(flows_2013_2017, 'flow_2013_2017')

    flows = flows_2013_2017.merge(flows_2018_2022)
    flows['IMPACT_code'] = category
    
    return flows

def get_prod_prices(category):

    prod_prices = pd.read_csv(f'../../OPSIS/Data/FAOSTAT/FAO_prod_prices/prod_prices_{category}.csv')
    prod_prices['IMPACT_code'] = category

    return prod_prices

def get_transport_data():
    
    transport_admin = pd.read_parquet('../../data/transport_data/global_lowest_transport_cost.parquet')
    transport_admin = transport_admin[['from_id', 'to_id', 'from_iso3', 'to_iso3', 'freight_USD_t', 'transport_USD_t', 'time_h', 'distance_km',
                           'border_USD_t', 'mode', 'trade_USD_t', 'customs_cost']]
    
    # adding the within-country transport files
    files = os.listdir('../../data/transport_data/Country_admin_transport/road_rail/')
    countries = list(set([f.split('.')[0].split('_')[-1] for f in files]))
    
    df_list = []
    
    for c in countries:
        
        transport_files = [f for f in files if c in f]
        transport_country = pd.concat([pd.read_parquet(f"../../data/transport_data/Country_admin_transport/road_rail/{f}") for f in transport_files])
        transport_country['to_id'] = transport_country.apply(lambda row: row['to_id_edge'].split('-')[0], axis=1)
        
        transport_country = transport_country[['from_id', 'to_id', 'from_iso3', 'to_iso3', 
               'transport_USD_t', 'time_h', 'distance_km', 'border_USD_t', 'mode']]
        transport_country['trade_USD_t'] = transport_country['transport_USD_t'] + transport_country['border_USD_t']
        transport_country['customs_cost'] = np.NaN
        transport_country['freight_USD_t'] = np.NaN
        
        transport_country = transport_country.sort_values(by=['from_id', 'to_id', 'transport_USD_t']).reset_index(drop=True)
        transport_country = transport_country.drop_duplicates(subset=['from_id', 'to_id'], keep='first')
        
        df_list.append(transport_country)
    
    transport_admin_country = pd.concat(df_list, ignore_index=True)
    
    maritime_incl_bulk = pd.read_parquet('../../data/transport_data/Country_admin_transport/maritime/domestic_maritime_transport_including_bulk.parquet')
    maritime_no_bulk = pd.read_parquet('../../data/transport_data/Country_admin_transport/maritime/domestic_maritime_transport_no_bulk.parquet')
    maritime_country = pd.concat([maritime_incl_bulk, maritime_no_bulk], ignore_index=True).drop_duplicates()
    maritime_country = maritime_country.sort_values(by=['from_id', 'to_id', 'transport_USD_t']).reset_index(drop=True)
    maritime_country = maritime_country.drop_duplicates(subset=['from_id', 'to_id'], keep='first')
    maritime_country = maritime_country[['from_id', 'to_id', 'from_iso3', 'to_iso3', 
               'transport_USD_t', 'time_h', 'distance_km', 'mode']]
    maritime_country['border_USD_t'] = 0
    maritime_country['trade_USD_t'] = maritime_country['transport_USD_t'] + maritime_country['border_USD_t']
    maritime_country['customs_cost'] = np.NaN
    maritime_country['freight_USD_t'] = np.NaN
    transport_admin_country = pd.concat([transport_admin_country, maritime_country], ignore_index=True)
    transport_admin_country = transport_admin_country.sort_values(by=['from_id', 'to_id', 'transport_USD_t']).reset_index(drop=True)
    transport_admin_country = transport_admin_country.drop_duplicates(subset=['from_id', 'to_id'], keep='first')
    
    transport_admin = pd.concat([transport_admin_country, transport_admin], ignore_index=True)
    transport_admin = transport_admin.sort_values(by=['from_id', 'to_id', 'transport_USD_t']).reset_index(drop=True)
    transport_admin = transport_admin.drop_duplicates(subset=['from_id', 'to_id'], keep='first').reset_index(drop=True)
    
    transport_admin.loc[transport_admin['customs_cost'].isna(), 'customs_cost'] = 0
    transport = transport_admin.groupby(['from_iso3', 'to_iso3'])[[
        'transport_USD_t', 'time_h', 'distance_km','border_USD_t', 'trade_USD_t', 'customs_cost']].mean().reset_index()
    
    return transport

def get_tariffs_data(transport, mm, codes):

    # import tariffs data - codes based on https://comtradeplus.un.org/ListOfReferences 
    mm = mm[(mm['hs6_rev2017'].isin(codes))]
    mm_country = pd.read_csv('../../data/Import_tariffs/Countries.csv')
    
    mm = mm.merge(mm_country, right_on='code', left_on='importer').drop(
        'code', axis=1).rename(columns={'iso3': 'to_iso3'})
    mm = mm.merge(mm_country, right_on='code', left_on='exporter').drop(
        'code', axis=1).rename(columns={'iso3': 'from_iso3'})
    mm = mm.drop(['importer', 'exporter', 'hs6_rev2017'], axis=1)
    mm = mm.groupby(['from_iso3','to_iso3']).mean()[['Pref_Applied_AVE']].reset_index()

    # combining transport and tariffs data 
    transport = transport.merge(mm, how='left')
    
    return transport

def get_demand_elas():
    demand_elas = pd.read_csv('../../OPSIS/Data/Future_production_demand_data/IMPACT-master/DriverAssumptions/Elasticities_Demand/ExogFDDmdElasH.csv', 
                              header=None)
    demand_elas.columns = ['IMPACT_code','Abbreviation','year','demand_elas']
    demand_elas['IMPACT_code'] = 'j'+demand_elas['IMPACT_code'].str[1:5]
    
    demand_elas = demand_elas[demand_elas['year']>=2020].reset_index(drop = True)
    demand_elas = demand_elas[demand_elas['year'].isin([2020,2025,2030,2035,2040,2045,2050])]
    ## add baseline ##
    demand_elas = demand_elas.merge(demand_elas[demand_elas['year']==2020][['Abbreviation','IMPACT_code','demand_elas']].copy().rename(
        columns = {'demand_elas':'demand_elas_baseline'}), on = ['Abbreviation','IMPACT_code'])
    demand_elas = demand_elas[demand_elas['IMPACT_code'].isin(['jwhea', 'jrice', 'jmaiz', 'jbarl',
                                                               'jmill', 'jsorg', 'jocer', 'jcass',
                                                               'jpota', 'jyams', 'jswpt', 'jorat',
                                                               'jvege', 'jbana', 'jplnt', 'jsubf', 
                                                               'jtemf', 'jbean', 'jchkp', 'jcowp',  
                                                               'jlent', 'jpigp', 'jopul', 'jsoyb', 
                                                               'jgrnd', 'jothr', 'jrpsd', 'jsnfl', 
                                                               'jtols', 'jpalm', 'jsugr'])]
    
    demand_elas = demand_elas[~demand_elas['Abbreviation'].isin(['ERI', 'GNQ', 'OSA', 'PSE'])]
    demand_elas_sugb = demand_elas[demand_elas['IMPACT_code']=='jsugr']
    demand_elas_sugb['IMPACT_code'] = 'jsugb'
    demand_elas.loc[demand_elas['IMPACT_code']=='jsugr', 'IMPACT_code'] = 'jsugc'
    demand_elas = pd.concat([demand_elas, demand_elas_sugb], axis=0, ignore_index=True)
    return demand_elas

def get_supply_elas():

    def _add_missing_regions(g, abbr):
        g = pd.DataFrame(abbr, columns=['Abbreviation']).merge(g, how='left')
        g.loc[g['IMPACT_code'].isna(), 'IMPACT_code'] = g[g['IMPACT_code'].notnull()]['IMPACT_code'].values[0]
        g.loc[g['supply_elas'].isna(), 'supply_elas'] = g['supply_elas'].mean()
        return g

    supply_elas = pd.read_csv('../../OPSIS/Data/Future_production_demand_data/IMPACT-master/DriverAssumptions/Elasticities_Supply/AreaElas.csv', 
                              header=None)
    supply_elas.columns = ['IMPACT_code','FPU','Irr','supply_elas']
    supply_elas = supply_elas[supply_elas['IMPACT_code'].isin(['jwhea', 'jrice', 'jmaiz', 'jbarl',
                                                               'jmill', 'jsorg', 'jocer', 'jcass',
                                                               'jpota', 'jyams', 'jswpt', 'jorat',
                                                               'jvege', 'jbana', 'jplnt', 'jsubf', 
                                                               'jtemf', 'jbean', 'jchkp', 'jcowp',  
                                                               'jlent', 'jpigp', 'jopul', 'jsoyb', 
                                                               'jgrnd', 'jothr', 'jrpsd', 'jsnfl', 
                                                               'jtols', 'jpalm', 'jsugb', 'jsugc'])]
    
    supply_elas = supply_elas.groupby(['IMPACT_code', 'FPU'])[['supply_elas']].mean().reset_index()
    
    basins = pd.read_excel('../../OPSIS/Data/Future_production_demand_data/IMPACT-master/DriverAssumptions/CorrespondenceFiles/Sets.xlsx',
                           sheet_name='Regions', header=3)
    basins = basins[['FPU.1', 'CTY.2']].rename(columns={'FPU.1': 'FPU', 'CTY.2': 'Abbreviation'})
    supply_elas = supply_elas.merge(basins)
    supply_elas = supply_elas[~supply_elas['Abbreviation'].isin(['ERI', 'GNQ', 'OSA', 'PSE'])]
    supply_elas = supply_elas.groupby(['IMPACT_code', 'Abbreviation'])[['supply_elas']].mean().reset_index()

    supply_elas = supply_elas.groupby('IMPACT_code').apply(lambda g: _add_missing_regions(
        g, supply_elas['Abbreviation'].unique())).reset_index(drop=True)

    return supply_elas

def combine_country_data(prod_prices, demand_elas, supply_elas, flows, FAO_area_codes):

    # aggregate flows by abbreviation, get imports and exports
    flows = flows.merge(FAO_area_codes, left_on='from_iso3', right_on='iso3').drop('iso3', axis=1).rename(
        columns={'Abbreviation': 'from_abbreviation'}).merge(FAO_area_codes, left_on='to_iso3', right_on='iso3').drop(
        'iso3', axis=1).rename(columns={'Abbreviation': 'to_abbreviation'})
    
    prod = flows.groupby('from_iso3')[['flow_2018_2022']].sum().reset_index().rename(
        columns={'from_iso3': 'iso3', 'flow_2018_2022': 'Production'})

    # aggregate production, yield, prod price by abbreviation
    df_country = prod_prices.merge(FAO_area_codes).drop('Production', axis=1).merge(prod)
    df_country['Area_dup'] = df_country['Area']
    df_country['Production_dup'] = df_country['Production']
    df_country.loc[df_country['Area_dup']==0, 'Area_dup'] = 0.01 # to avoid getting nulls when aggregating 
    df_country.loc[df_country['Production_dup']==0, 'Production_dup'] = 0.01 # to avoid getting nulls when aggregating 
    df_country['total_yield'] = df_country['Yield'] * df_country['Area_dup']
    df_country['total_price'] = df_country['Producer_price'] * df_country['Production_dup']

    df_country = df_country.groupby(['Abbreviation'])[['Area', 'Production', 
                                                      'total_yield', 'total_price',
                                                      'Area_dup', 'Production_dup']].sum().reset_index()
    df_country['Yield'] = df_country['total_yield'] / df_country['Area_dup']
    df_country['Producer_price'] = df_country['total_price'] / df_country['Production_dup']
    df_country = df_country.drop(['total_yield', 'total_price', 'Area_dup', 'Production_dup'], axis=1)

    # merge with demand and supply elasticities
    df_country = df_country.merge(demand_elas[demand_elas['year']==2020].drop(['year', 'demand_elas_baseline'], axis=1))
    df_country = df_country.merge(supply_elas)
    
    flows = flows.groupby(['from_abbreviation', 'to_abbreviation', 'IMPACT_code'])[['flow_2018_2022']].sum().reset_index()
    imports = flows[flows['from_abbreviation']!=flows['to_abbreviation']].groupby(
        ['to_abbreviation', 'IMPACT_code'])[['flow_2018_2022']].sum().reset_index().rename(
        columns={'to_abbreviation': 'Abbreviation', 'flow_2018_2022': 'imports'})
    exports = flows[flows['from_abbreviation']!=flows['to_abbreviation']].groupby(
        ['from_abbreviation', 'IMPACT_code'])[['flow_2018_2022']].sum().reset_index().rename(
        columns={'from_abbreviation': 'Abbreviation', 'flow_2018_2022': 'exports'})
    df_country = df_country.merge(imports).merge(exports)
    
    df_country['Consumption'] = df_country['Production'] + df_country['imports'] - df_country['exports'] 
    df_country = df_country.rename(columns={'Abbreviation': 'abbreviation',
                                            'Production': 'supply_q',
                                            'Consumption': 'demand_q',
                                            'imports': 'import_q',
                                            'exports': 'export_q',
                                            'Producer_price': 'prod_price_USD_t',
                                            'Yield': 'yield_t_ha'}) 
    df_country['domestic_q'] = df_country['supply_q'] - df_country['export_q']

    return df_country

def combine_bilateral_data(tariffs, flows, prod_prices, FAO_area_codes):
    
    tariffs = tariffs.merge(FAO_area_codes, left_on='from_iso3', right_on='iso3').drop(
        ['iso3', 'M49 Code'], axis=1).rename(columns={'Abbreviation': 'from_abbreviation'})
    tariffs = tariffs.merge(FAO_area_codes, left_on='to_iso3', right_on='iso3').drop(
        ['iso3', 'M49 Code'], axis=1).rename(columns={'Abbreviation': 'to_abbreviation'})

    df_bilateral = flows.merge(tariffs).fillna(0)

    # aggregate flows, trade_costs, adv by abbreviation pairs
    df_bilateral = df_bilateral.merge(prod_prices, left_on=['from_iso3', 'IMPACT_code'], right_on=['iso3', 'IMPACT_code']).drop('iso3', axis=1)
    df_bilateral['flow_dup'] = df_bilateral['flow_2018_2022']
    df_bilateral.loc[df_bilateral['flow_dup']==0, 'flow_dup'] = 0.01 # to avoid getting nulls when aggregating 
    df_bilateral['total_trade_cost'] = df_bilateral['trade_USD_t'] * df_bilateral['flow_dup']    
    df_bilateral['tariff_value'] = (df_bilateral['Producer_price'] + df_bilateral['trade_USD_t']) * df_bilateral['flow_dup']
    df_bilateral.loc[df_bilateral['tariff_value']==0, 'tariff_value'] = 0.0001 # to avoid getting nulls when aggregating 
    df_bilateral['total_tariff'] = (df_bilateral['Producer_price'] + df_bilateral['trade_USD_t']) * df_bilateral['Pref_Applied_AVE'] * df_bilateral['flow_dup']
    df_bilateral = df_bilateral.groupby(['from_abbreviation', 'to_abbreviation', 'IMPACT_code'])[[
        'flow_2013_2017', 'flow_2018_2022', 'total_trade_cost', 'total_tariff', 'flow_dup', 'tariff_value']].sum().reset_index()
    df_bilateral['trade_USD_t'] = df_bilateral['total_trade_cost'] / df_bilateral['flow_dup']
    df_bilateral['Pref_Applied_AVE'] = df_bilateral['total_tariff'] / df_bilateral['tariff_value']
    df_bilateral = df_bilateral.drop(['total_trade_cost', 'total_tariff', 'flow_dup', 'tariff_value'], axis=1)

    df_bilateral['trade_relationship'] = 0
    df_bilateral.loc[df_bilateral['flow_2018_2022']>1, 'trade_relationship'] = 1
    df_bilateral['trade_relationship_old'] = 0
    df_bilateral.loc[df_bilateral['flow_2013_2017']>1, 'trade_relationship_old'] = 1
    df_bilateral = df_bilateral.rename(columns={'Pref_Applied_AVE': 'adv',
                                                'flow_2013_2017': 'q_old',
                                                'flow_2018_2022': 'q_calib'})
    
    return df_bilateral

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
    FAO_area_codes = FAO_area_codes[~FAO_area_codes['iso3'].isin(['COK', 'KIR', 'MDV', 'MHL', 'MLT', 'NIU', 'STP', 'TKL', 'FRO'])] # removing countries with incomplete trade cost data
    transport = get_transport_data()
    mm_2019 = pd.read_csv('../../data/Import_tariffs/mm2019.csv', sep=';')
    demand_elas = get_demand_elas()
    supply_elas = get_supply_elas()
    
    for category in items_dict.keys():
        print(category)
        
        flows = get_flows(category)
        prod_prices = get_prod_prices(category) # in USD per tonne
        tariffs = get_tariffs_data(transport, mm_2019, items_dict[category])

        df_country = combine_country_data(prod_prices, demand_elas, supply_elas, flows, FAO_area_codes)
        df_bilateral = combine_bilateral_data(tariffs, flows, prod_prices, FAO_area_codes)
        
        # exporting country datasets
        df_country[['abbreviation', 'IMPACT_code', 'yield_t_ha', 'prod_price_USD_t',
                    'import_q', 'export_q', 'demand_q', 'supply_q', 'domestic_q', 'demand_elas', 'supply_elas'
                    ]].to_csv(f'../../OPSIS/Data/Trade_clearance_model/Input/Country_data/country_information_{category}.csv', index=False)

        # exporting bilateral datasets
        df_bilateral[['from_abbreviation', 'to_abbreviation', 'IMPACT_code', 'trade_USD_t', 'adv',
                      'q_calib', 'trade_relationship', 'q_old', 'trade_relationship_old' 
                     ]].to_csv(f'../../OPSIS/Data/Trade_clearance_model/Input/Trade_cost/bilateral_trade_cost_{category}.csv', index=False)
        