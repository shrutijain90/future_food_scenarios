# Usage: python -m future_trade.data_inputs.producer_prices

import pandas as pd
import numpy as np
import statsmodels.api as sm

from future_trade.data_inputs.balance_trade import get_area_codes

data_dir_prefix = '../../data/'

def get_prod(prod, item, year, FAO_area_codes):
    
    # preparing production matrix
    prod = prod[['Area Code (M49)', 'Item', 'Element', 'Year', 'Value']]
    prod = prod[(prod['Year']==year) & (prod['Item']==item) 
                 & (prod['Element'].isin(['Production', 'Area harvested']))].reset_index(drop=True)
    prod = prod.pivot(index=['Area Code (M49)', 'Item', 'Year'], columns='Element', values='Value').reset_index()
    prod = prod.rename(columns={'Area harvested': 'Area', 'Area Code (M49)': 'M49 Code'})
    
    
    prod['M49 Code'] = prod.apply(lambda row: int(row['M49 Code'][1:]), axis=1)
    prod = prod.merge(FAO_area_codes, how='right')
    prod = prod[['M49 Code', 'iso3', 'Region Name', 'Sub-region Name', 'Intermediate Region Name', 'Item', 'Area', 'Production']]
    prod = prod.fillna(0)
    prod['Item'] = item

    # adjustments to rice production
    if item=='Rice':
        prod['Production'] = prod['Production'] * 0.7 # scaling down to account for milling
    
    return prod

def split_prod(sua, prod, item, year, FAO_area_codes, proc):
    sua = sua[(sua['Element'].isin(['Feed', 'Food supply quantity (tonnes)', 'Loss', 'Other uses (non-food)',
                                    'Residuals', 'Seed', 'Processed', 'Tourist consumption']))
    & (sua['Year']==year) ].groupby(['Area', 'Area Code (M49)', 'Element', 'Item'])[['Value']].mean().reset_index().pivot(
        index=['Area', 'Area Code (M49)', 'Item'], columns='Element', values='Value').reset_index()

    sua = sua.rename(columns={'Food supply quantity (tonnes)': 'Food', 'Loss': 'Losses', 'Processed': 'Processing'})
    sua = sua.fillna(0)
    sua['Domestic supply quantity'] = sua['Feed'] + sua['Food'] + sua['Losses'] + sua['Seed'] + sua['Processing'] + sua['Other uses (non-food)'] \
    + sua['Tourist consumption'] + sua['Residuals']
    sua.loc[sua['Domestic supply quantity']<0, 'Domestic supply quantity'] = 0

    if item=='Olives':
        sua1 = sua[sua['Item']=='Olives']
        sua2 = sua[sua['Item']=='Olives preserved']
        sua = sua1[['Area Code (M49)', 'Processing']].merge(sua2[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        sua.loc[sua['Domestic supply quantity'].isna(), 'Domestic supply quantity'] = 0
        sua['Processing'] = sua['Processing'] - sua['Domestic supply quantity']
        sua = sua[['Area Code (M49)', 'Processing']].merge(sua1[['Area Code (M49)', 'Domestic supply quantity']], how='left')
    elif item=='Coconuts, in shell':
        sua1 = sua[sua['Item']=='Coconuts, in shell']
        sua2 = sua[sua['Item']=='Coconuts, desiccated']
        sua = sua1[['Area Code (M49)', 'Processing']].merge(sua2[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        sua.loc[sua['Domestic supply quantity'].isna(), 'Domestic supply quantity'] = 0
        sua['Processing'] = sua['Processing'] - sua['Domestic supply quantity']
        sua = sua[['Area Code (M49)', 'Processing']].merge(sua1[['Area Code (M49)', 'Domestic supply quantity']], how='left')
    elif item=='Groundnuts, excluding shelled':
        sua1 = sua[sua['Item']=='Groundnuts, excluding shelled']
        sua2 = sua[sua['Item'].isin(['Groundnuts, shelled', 'Prepared groundnuts'])]
        sua2 = sua2.groupby('Area Code (M49)')[['Domestic supply quantity']].sum().reset_index()
        sua = sua1[['Area Code (M49)', 'Processing']].merge(sua2[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        sua.loc[sua['Domestic supply quantity'].isna(), 'Domestic supply quantity'] = 0
        sua['Processing'] = sua['Processing'] - sua['Domestic supply quantity']
        sua = sua[['Area Code (M49)', 'Processing']].merge(sua1[['Area Code (M49)', 'Domestic supply quantity']], how='left')
    else:
        sua = sua[sua['Item']==item][['Area Code (M49)', 'Processing', 'Domestic supply quantity']]

    sua.loc[sua['Processing']<0, 'Processing'] = 0
    sua['Area Code (M49)'] = sua.apply(lambda row: int(row['Area Code (M49)'][1:]), axis=1)
    sua = sua.rename(columns={'Area Code (M49)': 'M49 Code'})
    sua = sua.merge(FAO_area_codes, how='right')
    sua = sua[['M49 Code', 'iso3', 'Domestic supply quantity', 'Processing']]
    sua = sua.fillna(0)
    sua['proc_prop'] = sua['Processing'] / sua['Domestic supply quantity']
    sua.loc[sua['Domestic supply quantity']==0, 'proc_prop'] = 0
    sua.loc[sua['proc_prop']>1, 'proc_prop'] = 1
    if proc==0:
        sua['proc_prop'] = 1 - sua['proc_prop']
    prod = prod.merge(sua[['iso3', 'M49 Code', 'proc_prop']])
    prod['Production'] = prod['Production'] * prod['proc_prop']
    prod['Area'] = prod['Area'] * prod['proc_prop']

    return prod

def get_prod_prices(prod_prices, item):

    def sel_years(df):
        if len(df)<3:
            return df # if less than 3 observations are available, keeping whatever's available
        return df.sort_values('Year')[-3:]

    prod_prices = prod_prices[(prod_prices['Unit']=='USD') & (prod_prices['Months']=='Annual value') 
    & (prod_prices['Year']>=2010)& (prod_prices['Item']==item)][[
        'Area Code (M49)', 'Item', 'Year', 'Value']].rename(columns={'Value': 'Producer Price (USD/tonne)'}).reset_index(drop=True)
    
    prod_prices['Area Code (M49)'] = prod_prices.apply(lambda row: int(row['Area Code (M49)'][1:]), axis=1)
    prod_prices = prod_prices.rename(columns={'Area Code (M49)': 'M49 Code'})
    
    prod_prices = prod_prices.groupby(['M49 Code']).apply(lambda g: sel_years(g)).reset_index(drop=True)
    
    prod_prices = prod_prices[['M49 Code', 'Item', 'Producer Price (USD/tonne)']].groupby(
        ['M49 Code', 'Item']).agg({'Producer Price (USD/tonne)': ['mean', 'count']}).reset_index()
    
    prod_prices.columns = prod_prices.columns.droplevel(0)
    prod_prices.reset_index(inplace=True, drop=True)
    prod_prices.columns = ['M49 Code', 'Item', 'Producer Price (USD/tonne)', 'num_years']
    
    prod_prices = prod_prices.merge(FAO_area_codes, how='right')
    prod_prices = prod_prices[['M49 Code', 'iso3', 'Region Name', 'Sub-region Name', 'Intermediate Region Name', 'Item', 'Producer Price (USD/tonne)', 'num_years']]
    prod_prices = prod_prices.fillna(0)
    prod_prices['Item'] = item

    return prod_prices

def get_gdp():
    gdp = pd.read_csv('../../data/FAOSTAT_A-S_E/Macro-Statistics_Key_Indicators_E_All_Data_(Normalized)/Macro-Statistics_Key_Indicators_E_All_Data_(Normalized).csv',
                        encoding='latin1')
    gdp = gdp[(gdp['Item']=='Gross Domestic Product') 
        & (gdp['Element']=='Value US$ per capita, 2015 prices')
        & (gdp['Year'].isin([2018, 2019, 2020, 2021, 2022]))
        ][['Area Code (M49)', 'Year', 'Value']].reset_index(drop=True)
    
    gdp['Area Code (M49)'] = gdp.apply(lambda row: int(row['Area Code (M49)'][1:]), axis=1)
    gdp = gdp.rename(columns={'Area Code (M49)': 'M49 Code', 'Value': 'GDP'})
    gdp = gdp[['M49 Code', 'GDP']].groupby(['M49 Code']).mean().reset_index()

    return gdp

def combine_data(prod, prod_prices, gdp, sua, item, FAO_area_codes, category):

    prod_list = []
    for year in [2018, 2019, 2020, 2021, 2022]:
        prod_year = get_prod(prod, item, year, FAO_area_codes)
    
        if item in ['Olives', 'Coconuts, in shell', 'Groundnuts, excluding shelled', 'Linseed', 'Hempseed', 'Sunflower seed', 'Safflower seed', 'Sesame seed']:
            if category in ['jsnfl', 'jtols']:
                prod_year = split_prod(sua, prod_year, item, year, FAO_area_codes, proc=1)
            else:
                prod_year = split_prod(sua, prod_year, item, year, FAO_area_codes, proc=0)
        prod_list.append(prod_year)

    prod = pd.concat(prod_list, axis=0, ignore_index=True)
    prod = prod[['iso3', 'Region Name', 'Sub-region Name', 'Intermediate Region Name', 'M49 Code', 'Item', 'Area', 'Production']].groupby(
        ['iso3', 'Region Name', 'Sub-region Name', 'Intermediate Region Name', 'M49 Code', 'Item']).mean().reset_index()

    prod_prices = get_prod_prices(prod_prices, item)
    df = prod.merge(gdp, how='left').merge(prod_prices, how='left').rename(columns={'Producer Price (USD/tonne)': 'Producer_price'})
    
    return df

def reg(df, n, type='log'):

    X_cols = [
        'log_GDP', 
        'log_prod',
        'Yield',
        'region_var'
    ]
    fml = "log_price ~ " + " + ".join(X_cols)
    
    print(f'{df.shape[0]} countries have production data')
    print(f"{df[df['Producer_price']>0].shape[0]} countries have producer price data")

    if df[df['Producer_price']>0].shape[0]<n:
        
        df['y_pred'] = 0
        return df.head(0)
        
    else:
        data = df[df['Producer_price']>0].sort_values(by='Production', ascending=False).head(n)
        
        if type=='log':
            mod = sm.OLS.from_formula(fml, data=data)
            mod2 = sm.OLS.from_formula(fml, data=df[df['Producer_price']>0])
        else:
            mod = sm.GLM.from_formula(fml, family=sm.families.Gamma(link=sm.families.links.Log()), data=data)
            mod2 = sm.GLM.from_formula(fml, family=sm.families.Gamma(link=sm.families.links.Log()), data=df[df['Producer_price']>0])
        res = mod.fit()
        res2 = mod2.fit()
    
        df2 = df[~df['region_var'].isin(data['region_var'].unique())]
        if len(df2)>0:
            df1 = df[df['region_var'].isin(data['region_var'].unique())]
            df1['y_pred'] = res.predict(df1[X_cols])
            df2_2 = df2[~df2['region_var'].isin(df[df['Producer_price']>0]['region_var'].unique())]
            if len(df2_2)>0:
                df2 = df2[df2['region_var'].isin(df[df['Producer_price']>0]['region_var'].unique())]
                df2['y_pred'] = res2.predict(df2[X_cols])
            else:
                df2['y_pred'] = res2.predict(df2[X_cols])
            df = pd.concat([df1, df2], axis=0, ignore_index=True)
        else:
            df['y_pred'] = res.predict(df[X_cols])
    
        if type=='log':
            df['y_pred'] = np.exp(df['y_pred'])

    return df

def gap_fill_price(df_gaps, df_prices):

    # fao cost averages by regions
    df_int_reg_avg = df_prices[['Intermediate Region Name', 'Producer_price']].groupby(['Intermediate Region Name']).mean().reset_index()
    df_sub_reg_avg = df_prices[['Sub-region Name', 'Producer_price']].groupby(['Sub-region Name']).mean().reset_index()
    df_reg_avg = df_prices[['Region Name', 'Producer_price']].groupby(['Region Name']).mean().reset_index()
    df_glo_avg = df_prices[['Producer_price']].mean()

    # filling nans with regional averages
    df_gaps = df_gaps.merge(df_int_reg_avg.rename(columns={'Producer_price': 'Int_Producer_price'}), how='left').merge(
        df_sub_reg_avg.rename(columns={'Producer_price': 'Sub_Producer_price'}), how='left').merge(
        df_reg_avg.rename(columns={'Producer_price': 'Reg_Producer_price'}), how='left')
    
    df_gaps['Glo_Producer_price'] = df_glo_avg.values[0]
    
    row_cond1 = (df_gaps['y_pred']==0) & (df_gaps['Int_Producer_price']>0)
    df_gaps.loc[row_cond1, 'y_pred'] = df_gaps[row_cond1]['Int_Producer_price']
    row_cond2 = (df_gaps['y_pred']==0) & (df_gaps['Sub_Producer_price']>0)
    df_gaps.loc[row_cond2, 'y_pred'] = df_gaps[row_cond2]['Sub_Producer_price']
    row_cond3 = (df_gaps['y_pred']==0) & (df_gaps['Reg_Producer_price']>0)
    df_gaps.loc[row_cond3, 'y_pred'] = df_gaps[row_cond3]['Reg_Producer_price']
    df_gaps.loc[df_gaps['y_pred']==0, 'y_pred'] = df_gaps[df_gaps['y_pred']==0]['Glo_Producer_price']

    return df_gaps.drop(['Int_Producer_price', 'Sub_Producer_price', 'Reg_Producer_price', 'Glo_Producer_price'], axis=1)

def gap_fill_yield(df_gaps, df_yields):

    # fao yield averages by regions
    df_int_reg_avg = df_yields[['Intermediate Region Name', 'Yield']].groupby(['Intermediate Region Name']).mean().reset_index()
    df_sub_reg_avg = df_yields[['Sub-region Name', 'Yield']].groupby(['Sub-region Name']).mean().reset_index()
    df_reg_avg = df_yields[['Region Name', 'Yield']].groupby(['Region Name']).mean().reset_index()
    df_glo_avg = df_yields[['Yield']].mean()

    # filling nans with regional averages
    df_gaps = df_gaps.merge(df_int_reg_avg.rename(columns={'Yield': 'Int_Yield'}), how='left').merge(
        df_sub_reg_avg.rename(columns={'Yield': 'Sub_Yield'}), how='left').merge(
        df_reg_avg.rename(columns={'Yield': 'Reg_Yield'}), how='left')
    
    df_gaps['Glo_Yield'] = df_glo_avg.values[0]
    
    row_cond1 = (df_gaps['Yield']==0) & (df_gaps['Int_Yield']>0)
    df_gaps.loc[row_cond1, 'Yield'] = df_gaps[row_cond1]['Int_Yield']
    row_cond2 = (df_gaps['Yield']==0) & (df_gaps['Sub_Yield']>0)
    df_gaps.loc[row_cond2, 'Yield'] = df_gaps[row_cond2]['Sub_Yield']
    row_cond3 = (df_gaps['Yield']==0) & (df_gaps['Reg_Yield']>0)
    df_gaps.loc[row_cond3, 'Yield'] = df_gaps[row_cond3]['Reg_Yield']
    df_gaps.loc[df_gaps['Yield']==0, 'Yield'] = df_gaps[df_gaps['Yield']==0]['Glo_Yield']

    return df_gaps.drop(['Int_Yield', 'Sub_Yield', 'Reg_Yield', 'Glo_Yield'], axis=1)

# returns gap filled producer prices in USD/tonne
def reg_predict(df):
    # if either of production or area is 0, make sure the other is 0 also
    df.loc[df['Production']==0, 'Area'] = 0
    df.loc[df['Area']==0, 'Production'] = 0
    
    # first converting to usd/ha as the regression does better on usd/ha instead of usd/tonne
    df['Yield'] = df['Production'] / df['Area']
    df['Yield'] = df['Yield'].fillna(0)
    df['Producer_price'] = df['Producer_price'] * df['Yield']
    
    # subsetting to countries which have production 
    df_sub = df.copy()
    df_sub = df_sub[(df_sub['Production']>0) & ((df_sub['Area']>0))].reset_index(drop=True)
    df_sub = df_sub[df_sub['GDP'].notnull()].reset_index(drop=True)
    
    df_sub['log_GDP'] = np.log(df_sub['GDP'])
    df_sub['log_prod'] = np.log(df_sub['Production'])
    df_sub['log_price'] = np.log(df_sub['Producer_price'])
    df_sub['log_area'] = np.log(df_sub['Area'])
    df_sub['GDP2'] = df_sub['GDP'].pow(2)
    df_sub['log_prod2'] = df_sub['log_prod'].pow(2)
    df_sub['region_var'] = df_sub['Region Name']
    
    n = 25
    # if there is producer price information for at least n countries, 
    # use it to predict for the remaining countries that produce this item 
    # this can leave some gaps if there are no countries with producer price info in a specific region
    df_sub = reg(df_sub, n, type='log')
    df = df.merge(df_sub[['iso3', 'M49 Code', 'y_pred']], how='left')
    df.loc[df['y_pred']> 3 * df['Producer_price'].max(), 'y_pred'] = 0 # in very very few cases, price explodes if yields in prediction countries are much higher than yields in countries used to fit the regression. replacing those with regional means
    df['y_pred'] = df['y_pred'].fillna(0)
    
    # gap fill based on regional averages 
    # if less than 25 countries have price info, or if there is no price info for certain regions
    # we also want some price info for non-producing countries - obtaining that using gap filling as well
    df = gap_fill_price(df, df[df['Producer_price']>0])
    df.loc[df['Producer_price']==0, 'Producer_price'] = df[df['Producer_price']==0]['y_pred']
    df = df.drop('y_pred', axis=1)
    
    # convert back to usd/tonne
    # first gap fill yield for non-producing countries
    df = gap_fill_yield(df, df[df['Yield']>0])
    df['Producer_price'] = df['Producer_price'] / df['Yield']

    return df

if __name__ == '__main__':

    items_dict = {              
                  # wheat
                  'jwhea': ['Wheat'],
                  
                  # rice
                  'jrice': ['Rice'],
                  
                  # maize
                  'jmaiz': ['Maize (corn)'],
                  
                  # othr_grains
                  'jbarl': ['Barley'],
                  'jmill': ['Millet'], 
                  'jsorg': ['Sorghum'], 
                  'jocer': ['Rye', 'Oats', 'Buckwheat', 'Quinoa', 'Canary seed', 'Fonio', 'Mixed grain', 'Triticale', 'Cereals n.e.c.'], 
                  
                  # roots
                  'jcass': ['Cassava, fresh'], 
                  'jpota': ['Potatoes'], 
                  'jswpt': ['Sweet potatoes'],
                  'jyams': ['Yams'],
                  'jorat': ['Taro', 'Edible roots and tubers with high starch or inulin content, n.e.c., fresh'],
                  
                  # vegetables
                  'jvege': ['Artichokes', 'Asparagus', 'Broad beans and horse beans, green', 'Cabbages', 'Carrots and turnips', 'Cauliflowers and broccoli',
                            'Chillies and peppers, green (Capsicum spp. and Pimenta spp.)', 'Cucumbers and gherkins', 'Eggplants (aubergines)', 'Green corn (maize)',
                            'Green garlic', 'Leeks and other alliaceous vegetables', 'Lettuce and chicory', 'Okra',
                            'Onions and shallots, dry (excluding dehydrated)', 'Onions and shallots, green', 'Other beans, green', 'Other vegetables, fresh n.e.c.',
                            'Peas, green', 'Pumpkins, squash and gourds', 'Spinach', 'String beans', 'Tomatoes'], # 'Mushrooms and truffles'
                  
                  # fruits
                  'jbana': ['Bananas'], 
                  'jplnt': ['Plantains and cooking bananas'], 
                  'jsubf': ['Apricots', 'Avocados', 'Cantaloupes and other melons', 'Dates', 'Figs', 
                            'Kiwi fruit', 'Lemons and limes', 'Locust beans (carobs)', 'Mangoes, guavas and mangosteens', 
                            'Oranges', 'Other citrus fruit, n.e.c.', 'Other fruits, n.e.c.', 'Other tropical fruits, n.e.c.', 'Papayas', 
                            'Pineapples', 'Pomelos and grapefruits', 'Tangerines, mandarins, clementines', 
                            'Watermelons', 'Coconuts, in shell'], # 'Cashewapple'
                  'jtemf': ['Apples', 'Grapes', 'Blueberries', 'Cherries', 'Cranberries', 'Currants', 'Gooseberries', 
                            'Other berries and fruits of the genus vaccinium n.e.c.', 'Other pome fruits', 'Other stone fruits', 
                            'Peaches and nectarines', 'Pears', 'Persimmons', 'Plums and sloes', 'Quinces', 'Raspberries', 
                            'Sour cherries', 'Strawberries', 'Olives'], 
                  
                  # legumes
                  'jbean': ['Bambara beans, dry', 'Beans, dry', 'Broad beans and horse beans, dry'], 
                  'jchkp': ['Chick peas, dry'],
                  'jcowp': ['Cow peas, dry'],
                  'jlent': ['Lentils, dry'], 
                  'jpigp': ['Pigeon peas, dry'], 
                  'jopul': ['Lupins', 'Other pulses n.e.c.', 'Peas, dry', 'Vetches'], 
                  
                  # soybeans
                  'jsoyb': ['Soya beans'],
                  
                  # nuts_seeds
                  'jgrnd': ['Groundnuts, excluding shelled'], 
                  'jothr': ['Almonds, in shell', 'Cashew nuts, in shell', 'Chestnuts, in shell', 'Hazelnuts, in shell', 
                            'Other nuts (excluding wild edible nuts and groundnuts), in shell, n.e.c.', 'Pistachios, in shell', 'Walnuts, in shell', 
                            'Linseed', 'Sunflower seed', 'Safflower seed', 'Poppy seed', 'Sesame seed'], # 'Brazil nuts, in shell', 'Hempseed'
                  
                  
                  # oil_veg
                  'jrpsd': ['Rape or colza seed'], 
                  'jsnfl': ['Sunflower seed'], 
                  'jtols': ['Groundnuts, excluding shelled', 'Linseed', 'Safflower seed', 'Sesame seed',
                            'Castor oil seeds', 'Coconuts, in shell', 'Mustard seed', 'Olives'], # 'Cotton seed', 'Hempseed'
                  
                  # oil_palm
                  'jpalm': ['Oil palm fruit'], 
                  
                  # sugar
                  'jsugb': ['Sugar beet'], 
                  'jsugc': ['Sugar cane']          
                  
    }

    prod = pd.read_csv(f'{data_dir_prefix}FAOSTAT_A-S_E/Production_Crops_Livestock_E_All_Data_(Normalized)/Production_Crops_Livestock_E_All_Data_(Normalized).csv',
                       encoding='latin1')
    prod_prices = pd.read_csv(f'{data_dir_prefix}FAOSTAT_A-S_E/Prices_E_All_Data_(Normalized)/Prices_E_All_Data_(Normalized).csv',
                              encoding='latin1')
    sua = pd.read_csv('../../data/FAOSTAT_A-S_E/SUA_Crops_Livestock_E_All_Data_(Normalized)/SUA_Crops_Livestock_E_All_Data_(Normalized).csv',
                          encoding='latin1', low_memory=False)

    FAO_area_codes = get_area_codes()
    gdp = get_gdp()

    for category in items_dict.keys():
        print(category)
        items = items_dict[category]
        df_category_list = []
        
        for item in items:
            print(item)
            df = combine_data(prod, prod_prices, gdp, sua, item, FAO_area_codes, category)
            print(df.shape)
            df = reg_predict(df)
            df_category_list.append(df)

        df_category = pd.concat(df_category_list)


        # aggregate by category
        df_category['Area_dup'] = df_category['Area']
        df_category['Production_dup'] = df_category['Production']
        df_category.loc[df_category['Area_dup']==0, 'Area_dup'] = 0.01 # to avoid getting nulls when aggregating by category
        df_category.loc[df_category['Production_dup']==0, 'Production_dup'] = 0.01 # to avoid getting nulls when aggregating by cateogory
        df_category['total_yield'] = df_category['Yield'] * df_category['Area_dup']
        df_category['total_price'] = df_category['Producer_price'] * df_category['Production_dup']

        df_category = df_category.groupby(['iso3'])[['Area', 'Production', 
                                                     'total_yield', 'total_price',
                                                     'Area_dup', 'Production_dup']].sum().reset_index()
        df_category['Yield'] = df_category['total_yield'] / df_category['Area_dup']
        df_category['Producer_price'] = df_category['total_price'] / df_category['Production_dup']
        df_category = df_category.drop(['total_yield', 'total_price', 'Area_dup', 'Production_dup'], axis=1)
        df_category.to_csv(f'../../OPSIS/Data/FAOSTAT/FAO_prod_prices/prod_prices_{category}.csv', index=False)
    