# Usage: python -m future_trade.data_inputs.align_regions

import pandas as pd
import numpy as np

if __name__ == '__main__':

    regions = pd.read_excel('../../OPSIS/Data/Future_production_demand_data/EAT_Lancet_release.xlsx', 
                      sheet_name='regions')
    countries = pd.read_excel('../../OPSIS/Data/Country_group/Country_classification_UNSD.xlsx')

    regions = regions.merge(countries[['Country or Area', 'iso3', 'M49 Code', 
                                'Intermediate Region Name', 'Sub-region Name', 'Region Name']], 
                left_on='Abbreviation', right_on='iso3', how='outer')

    regions.loc[regions['iso3'].isin(['EST', 'LVA', 'LTU']), 'Abbreviation'] = 'BLT'
    regions.loc[regions['iso3'].isin(['EST', 'LVA', 'LTU']), 'Region or country'] = 'Baltic States'

    regions.loc[regions['iso3'].isin(['BEL', 'LUX']), 'Abbreviation'] = 'BLX'
    regions.loc[regions['iso3'].isin(['BEL', 'LUX']), 'Region or country'] = 'Belgium and Luxembourg'

    regions.loc[regions['iso3'].isin(['CHN', 'HKG', 'MAC', 'TWN']), 'Abbreviation'] = 'CHM'
    regions.loc[regions['iso3'].isin(['CHN', 'HKG', 'MAC', 'TWN']), 'Region or country'] = 'China'

    regions.loc[regions['iso3'].isin(['CHE', 'LIE']), 'Abbreviation'] = 'CHP'
    regions.loc[regions['iso3'].isin(['CHE', 'LIE']), 'Region or country'] = 'Switzerland'

    regions.loc[regions['iso3'].isin(['AIA', 'ATG', 'ABW', 'BHS', 'BRB', 'BES', 'VGB', 'CYM', 'CUW', 'DMA', 
                                    'GRD', 'GLP', 'MTQ', 'MSR', 'PRI', 'BLM', 'KNA', 'LCA', 'MAF', 'VCT',
                                    'SXM', 'TTO', 'TCA', 'VIR']), 'Abbreviation'] = 'CRB'
    regions.loc[regions['iso3'].isin(['AIA', 'ATG', 'ABW', 'BHS', 'BRB', 'BES', 'VGB', 'CYM', 'CUW', 'DMA', 
                                    'GRD', 'GLP', 'MTQ', 'MSR', 'PRI', 'BLM', 'KNA', 'LCA', 'MAF', 'VCT',
                                    'SXM', 'TTO', 'TCA', 'VIR']), 'Region or country'] = 'Other Caribbean'

    regions.loc[regions['iso3'].isin(['FIN', 'ALA']), 'Abbreviation'] = 'FNP'
    regions.loc[regions['iso3'].isin(['FIN', 'ALA']), 'Region or country'] = 'Finland'

    regions.loc[regions['iso3'].isin(['FRA', 'MCO']), 'Abbreviation'] = 'FRP'
    regions.loc[regions['iso3'].isin(['FRA', 'MCO']), 'Region or country'] = 'France'

    regions.loc[regions['iso3'].isin(['GUF', 'GUY', 'SUR']), 'Abbreviation'] = 'GSA'
    regions.loc[regions['iso3'].isin(['GUF', 'GUY', 'SUR']), 'Region or country'] = 'Guyanas South America'

    regions.loc[regions['iso3'].isin(['ITA', 'VAT', 'MLT', 'SMR']), 'Abbreviation'] = 'ITP'
    regions.loc[regions['iso3'].isin(['ITA', 'VAT', 'MLT', 'SMR']), 'Region or country'] = 'Italy'

    regions.loc[regions['iso3'].isin(['MAR']), 'Abbreviation'] = 'MOR'
    regions.loc[regions['iso3'].isin(['MAR']), 'Region or country'] = 'Morocco'

    regions.loc[regions['iso3'].isin(['STP', 'CPV', 'SHN', 'BVT', 'SJM',	
                                    'FLK', 'SGS', 'BMU', 'SPM']), 'Abbreviation'] = 'OAO'
    regions.loc[regions['iso3'].isin(['STP', 'CPV', 'SHN', 'BVT', 'SJM',
                                    'FLK', 'SGS', 'BMU', 'SPM']), 'Region or country'] = 'Other Atlantic Ocean'

    regions.loc[regions['iso3'].isin(['SRB', 'MKD', 'MNE', 'BIH']), 'Abbreviation'] = 'OBN'
    regions.loc[regions['iso3'].isin(['SRB', 'MKD', 'MNE', 'BIH']), 'Region or country'] = 'Other Balkans'

    regions.loc[regions['iso3'].isin(['IOT', 'COM', 'ATF', 'MUS', 'MYT', 'REU', 'SYC', 'MDV']), 'Abbreviation'] = 'OIO'
    regions.loc[regions['iso3'].isin(['IOT', 'COM', 'ATF', 'MUS', 'MYT', 'REU', 'SYC', 'MDV']), 'Region or country'] = 'Other Indian Ocean'

    regions.loc[regions['iso3'].isin(['CXR', 'CCK', 'HMD', 'NFK', 'NCL', 'GUM', 
                                    'KIR', 'MHL', 'FSM','NRU', 'MNP', 'PLW', 
                                    'UMI', 'ASM', 'COK', 'PYF', 'NIU', 'PCN',
                                    'WSM', 'TKL', 'TON', 'TUV', 'WLF']), 'Abbreviation'] = 'OPO'
    regions.loc[regions['iso3'].isin(['CXR', 'CCK', 'HMD', 'NFK', 'NCL', 'GUM', 
                                    'KIR', 'MHL', 'FSM','NRU', 'MNP', 'PLW', 
                                    'UMI', 'ASM', 'COK', 'PYF', 'NIU', 'PCN',
                                    'WSM', 'TKL', 'TON', 'TUV', 'WLF']), 'Region or country'] = 'Other Pacific Ocean'

    regions.loc[regions['iso3'].isin(['BRN', 'SGP']), 'Abbreviation'] = 'OSA'
    regions.loc[regions['iso3'].isin(['BRN', 'SGP']), 'Region or country'] = 'Other Southeast Asia'

    regions.loc[regions['iso3'].isin(['BHR', 'KWT', 'OMN', 'QAT', 'ARE']), 'Abbreviation'] = 'RAP'
    regions.loc[regions['iso3'].isin(['BHR', 'KWT', 'OMN', 'QAT', 'ARE']), 'Region or country'] = 'Rest of Arab Peninsula'

    regions.loc[regions['iso3'].isin(['ESP', 'AND', 'GIB']), 'Abbreviation'] = 'SPP'
    regions.loc[regions['iso3'].isin(['ESP', 'AND', 'GIB']), 'Region or country'] = 'Spain'

    regions.loc[regions['iso3'].isin(['GBR', 'GGY', 'JEY', 'IMN']), 'Abbreviation'] = 'UKP'
    regions.loc[regions['iso3'].isin(['GBR', 'GGY', 'JEY', 'IMN']), 'Region or country'] = 'United Kingdom'

    regions.loc[regions['iso3'].isin(['FRO']), 'Abbreviation'] = 'DNK'
    regions.loc[regions['iso3'].isin(['FRO']), 'Region or country'] = 'Denmark'

    regions.loc[regions['iso3'].isin(['BVT']), 'Abbreviation'] = 'NOR'
    regions.loc[regions['iso3'].isin(['BVT']), 'Region or country'] = 'Norway'

    regions = regions[(regions['iso3'].notnull()) & (regions['Abbreviation'].notnull())].reset_index(drop=True)
    
    regions.to_csv('../../OPSIS/Data/Country_group/regions.csv', index=False)