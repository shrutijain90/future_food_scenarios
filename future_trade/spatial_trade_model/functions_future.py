import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.mpec import *
import math
import datetime
import logging
from future_trade.spatial_trade_model.functions_general import *

#### SHOCK MODEL ######
def process_final_output(model, year_select, crop_code, SSP,scen,error, error_scale):
    ### extract data
    trade = pd.DataFrame(pd.Series(model.trade3.extract_values())).reset_index().rename(columns={'level_0':'from_abbreviation','level_1':'to_abbreviation',0:'trade'})

    ### adjust
    trade['trade'] = np.where(trade['trade']<= error/error_scale, 0, trade['trade'])
    trade['year'] = year_select
    trade['IMPACT_code'] = crop_code
    trade['SSP'] = SSP
    trade['diet_scn'] = scen[0]
    trade['kcal_scn'] = scen[1]
    trade['RCP'] = scen[2]
    trade['lib_scn'] = scen[3]
    ### supply
    supply = trade.groupby(['from_abbreviation'])['trade'].sum().reset_index().rename(
        columns={'from_abbreviation':'abbreviation','trade':'supply'}).set_index(['abbreviation'])
    demand = trade.groupby(['to_abbreviation'])['trade'].sum().reset_index().rename(
        columns={'to_abbreviation':'abbreviation','trade':'demand'}).set_index(['abbreviation'])

    ## domestic supply
    domestic_supply =  trade[trade['from_abbreviation']==trade['to_abbreviation']].groupby(['from_abbreviation'])['trade'].sum().reset_index().rename(columns = {'from_abbreviation':'abbreviation','trade':'dom_supply'}).set_index(['abbreviation'])
    import_supply =  trade[trade['from_abbreviation']!=trade['to_abbreviation']].groupby(['to_abbreviation'])['trade'].sum().reset_index().rename(columns = {'to_abbreviation':'abbreviation','trade':'import'}).set_index(['abbreviation'])
    export_supply =  trade[trade['from_abbreviation']!=trade['to_abbreviation']].groupby(['from_abbreviation'])['trade'].sum().reset_index().rename(columns = {'from_abbreviation':'abbreviation','trade':'export'}).set_index(['abbreviation'])

    ## prodprice, conprice
    prodprice = pd.DataFrame(pd.Series(model.prodprice3.extract_values())).reset_index().rename(columns={'index':'abbreviation',0:'prodprice'}).set_index(['abbreviation'])
    conprice = pd.DataFrame(pd.Series(model.conprice3.extract_values())).reset_index().rename(columns={'index':'abbreviation',0:'conprice'}).set_index(['abbreviation'])

    ### producer and consumer surplus ###
    B_value = pd.DataFrame(pd.Series(model.B.extract_values())).reset_index().rename(columns={'index':'abbreviation',0:'B_value'}).set_index(['abbreviation'])
    D_value = pd.DataFrame(pd.Series(model.D.extract_values())).reset_index().rename(columns={'index':'abbreviation',0:'D_value'}).set_index(['abbreviation'])


    ### merge together 
    ### Q is in 1000t units. since B and D were calibrated to the 1000t units, their units are USD/(1000 t)^2. 
    ### Hence the two cancel out in the products resulting in USD units, which divided by 1e6 would be in million USD units.
    country_output = pd.concat([supply, demand, domestic_supply, import_supply, export_supply, prodprice, conprice, B_value, D_value], axis = 1)
    country_output['con_surplus_mUSD'] = (country_output['B_value']*country_output['demand']**2)/(2*1e6)
    country_output['prod_surplus_mUSD'] = (country_output['D_value']*country_output['supply']**2)/(2*1e6)

    ## add info ##
    country_output['year'] = year_select
    country_output['IMPACT_code'] = crop_code
    country_output['SSP'] = SSP
    country_output['diet_scn'] = scen[0]
    country_output['kcal_scn'] = scen[1]
    country_output['RCP'] = scen[2]
    country_output['lib_scn'] = scen[3]

    return trade, country_output.reset_index()


def read_calibration_output(calibration_output_path, crop_code):
    #### read calibration files as dictionaries
    trade = pd.read_csv(calibration_output_path+'trade_calibration_'+crop_code+'.csv', header=None, index_col=[0,1]).squeeze().to_dict()
    prodprice = pd.read_csv(calibration_output_path+'prodprice_calibration_'+crop_code+'.csv', header=None, index_col=[0]).squeeze().to_dict()
    conprice = pd.read_csv(calibration_output_path+'conprice_calibration_'+crop_code+'.csv', header=None, index_col=[0]).squeeze().to_dict()
    tc = pd.read_csv(calibration_output_path+'tc_calibration_'+crop_code+'.csv', header=None, index_col=[0,1]).squeeze().to_dict()
    calib_constant = pd.read_csv(calibration_output_path+'calib_calibration_'+crop_code+'.csv', header=None, index_col=[0,1]).squeeze().to_dict()

    return trade, prodprice, conprice, tc, calib_constant

def read_calibration_output_future(calibration_output_path, country_output,trade_output, crop_code, factor_error, error):
    ### add error and round ###
    trade_output['trade'] = np.round(np.where(trade_output['trade']<=error, error, trade_output['trade']), factor_error)

    #### read calibration files as dictionaries
    trade = trade_output.set_index(['from_abbreviation','to_abbreviation'])['trade'].squeeze().to_dict()
    prodprice = country_output[['abbreviation','prodprice']].set_index('abbreviation').squeeze().to_dict()
    conprice = country_output[['abbreviation','conprice']].set_index('abbreviation').squeeze().to_dict()

    ### same as original input ###
    tc = pd.read_csv(calibration_output_path+'tc_calibration_'+crop_code+'.csv', header=None, index_col=[0,1]).squeeze().to_dict()
    calib_constant = pd.read_csv(calibration_output_path+'calib_calibration_'+crop_code+'.csv', header=None, index_col=[0,1]).squeeze().to_dict()

    return trade, prodprice, conprice, tc, calib_constant

def update_country_dict(country_dict, country_output, year_select):
    ### extract shock a make dict ###
    new_demand = country_output[['abbreviation','demand']].set_index('abbreviation').squeeze().to_dict()
    new_supply = country_output[['abbreviation','supply']].set_index('abbreviation').squeeze().to_dict()
    demand_elas = country_output[['abbreviation',f'scaling_elas_{year_select}']].set_index('abbreviation').squeeze().to_dict()
    new_prodprice = country_output[['abbreviation','prodprice']].set_index('abbreviation').squeeze().to_dict()

    #### supply adjustment  ##
    for key, value in country_dict.supply.items():
        country_dict.supply[key] = new_supply[key]

    ### demand adjustment ###
    for key, value in country_dict.demand.items():
        country_dict.demand[key] = new_demand[key]

    ### demand elasticity ###
    for key, value in country_dict.demand_elas.items():
        country_dict.demand_elas[key] = country_dict.demand_elas[key]*demand_elas[key]

    ### prodprice adjustment ###
    for key, value in country_dict.demand.items():
        country_dict.production_cost[key] = new_prodprice[key]

    return country_dict

def update_bilateral_dict(bilateral_dict, trade_output):
    ### extract shock a make dict ###
    new_trade_old = trade_output.set_index(['from_abbreviation','to_abbreviation'])['trade'].squeeze().to_dict()
    new_trade_binary = trade_output.set_index(['from_abbreviation','to_abbreviation'])['trade_binary'].squeeze().to_dict()

    #### trade value  ##
    for key, value in bilateral_dict.trade_old.items():
        bilateral_dict.trade_old[key] = new_trade_old[key]

    ### trade binary ###
    for key, value in bilateral_dict.trade_binary.items():
        bilateral_dict.trade_binary[key] = new_trade_binary[key]

    return bilateral_dict

def create_scaling_parameters(year, crop_code, scen, future_demand_elas, future_demand_scaling, future_supply_scaling):
    
    demand_elas_scaling = future_demand_elas[(future_demand_elas['year']==year)
                                            & (future_demand_elas['IMPACT_code']==crop_code)][['abbreviation','scaling_factor_demand_elas']]
    demand_scaling = future_demand_scaling[(future_demand_scaling['year']==year)
                                            & (future_demand_scaling['diet_scn']==scen[0])
                                            & (future_demand_scaling['kcal_scn']==scen[1])][['abbreviation', 'scaling_factor_demand']]
    supply_scaling = future_supply_scaling[(future_supply_scaling['year']==year)
                                            & (future_supply_scaling['RCP']==scen[2])][['abbreviation','scaling_factor_yield', 'scaling_factor_supply']]

    ### scaling ##
    demand_elas_scaling = demand_elas_scaling.rename(columns={'scaling_factor_demand_elas':f'scaling_elas_{year}'})
    demand_scaling = demand_scaling.rename(columns={'scaling_factor_demand': f'scaling_demand_{year}'})
    supply_scaling = supply_scaling.rename(columns={'scaling_factor_supply': f'scaling_supply_{year}',
                                                    'scaling_factor_yield': f'scaling_yield_{year}'})
    model_scaling = demand_elas_scaling.merge(demand_scaling, how='outer').merge(supply_scaling, how='outer').fillna(1)
    return model_scaling

def read_update_future_model_data(model_output, crop_code, year_select, SSP, scen, future_demand_elas, supply_elas, future_demand_scaling, future_supply_scaling):
    
    ### read model output ##
    country_output = pd.read_csv(f'{model_output}Country_output/country_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{scen[3]}_{year_select-5}_{crop_code}.csv')
    country_output_2020 = pd.read_csv(f'{model_output}Country_output/country_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{scen[3]}_2020_{crop_code}.csv')
    country_output_2020['dom_share_supply'] = country_output_2020['dom_supply']/country_output_2020['supply']
    country_output_2020['dom_share_demand'] = country_output_2020['dom_supply']/country_output_2020['demand']

    ### extract the scaling ##
    model_scaling_year = create_scaling_parameters(year_select, crop_code, scen, future_demand_elas, future_demand_scaling, future_supply_scaling)
    model_scaling_year_previous = create_scaling_parameters(year_select-5, crop_code, scen, future_demand_elas, future_demand_scaling, future_supply_scaling)
    scaling_parameters = model_scaling_year.merge(model_scaling_year_previous, on='abbreviation')

    ### merge to original input at t-5 ##
    country_output = country_output.merge(scaling_parameters, how='left', on='abbreviation').replace(np.nan, 1).merge(supply_elas, on='abbreviation')

    ### scaling supply, demand, prodprice using growth factor ##
    country_output['supply'] = country_output['supply']*(country_output[f'scaling_supply_{year_select}']/country_output[f'scaling_supply_{year_select-5}'])
    country_output['dom_supply'] = country_output['dom_supply']*(country_output[f'scaling_supply_{year_select}']/country_output[f'scaling_supply_{year_select-5}'])

    country_output['demand'] = country_output['demand']*(country_output[f'scaling_demand_{year_select}']/country_output[f'scaling_demand_{year_select-5}'])
    country_output['import'] = country_output['import']*(country_output[f'scaling_demand_{year_select}']/country_output[f'scaling_demand_{year_select-5}'])

    ### supply elas in denominator here!!!! (as supply elas = perc change in quantity supplied wrt perc change in price)
    country_output['prodprice'] = country_output['prodprice']*(1+((1/country_output['supply_elas']) * ((country_output[f'scaling_supply_{year_select}']/country_output[f'scaling_supply_{year_select-5}'])-1)))/(country_output[f'scaling_yield_{year_select}']/country_output[f'scaling_yield_{year_select-5}'])

    return country_output, country_output_2020

def get_calibrated_demand_supply_2020(crop_code, calibration_output_path, error, error_scale):
    trade = pd.read_csv(calibration_output_path+'trade_calibration_'+crop_code+'.csv', header=None)
    trade.columns = ['from_abbreviation', 'to_abbreviation', 'trade']
    supply = trade.groupby(['from_abbreviation'])['trade'].sum().reset_index().rename(
            columns={'from_abbreviation':'abbreviation','trade':'supply'})
    supply['supply'] = supply['supply'] + (error/error_scale)*len(supply.supply)
    supply = supply.set_index(['abbreviation']).squeeze().to_dict()
    
    demand = trade.groupby(['to_abbreviation'])['trade'].sum().reset_index().rename(
            columns={'to_abbreviation':'abbreviation','trade':'demand'})
    demand['demand'] = demand['demand'] + (error/error_scale)*len(demand.demand)
    demand = demand.set_index(['abbreviation']).squeeze().to_dict()
    
    return demand, supply

def calculate_historical_trade_shares(model_output, crop_code, SSP, scen, target_SD):
    """
    Calculate historical trade shares from 2020 solver output.
    Returns dictionary with (i,j) keys and trade share values.
    """
    # Read 2020 trade output 
    trade_2020 = pd.read_csv(f'{model_output}Trade_output/trade_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{scen[3]}_2020_{crop_code}.csv')
    
    # Calculate total demand for each importing country from trade flows
    demand_total = trade_2020.groupby(['to_abbreviation'])['trade'].sum().reset_index().rename(
        columns={'trade':'total_demand'})
    
    # Merge with trade data to calculate shares
    trade_with_demand = trade_2020.merge(demand_total, on='to_abbreviation', how='left')
    
    # Calculate trade share for each bilateral relationship
    trade_with_demand['trade_share'] = trade_with_demand['trade'] / trade_with_demand['total_demand']
    trade_with_demand.loc[trade_with_demand['trade_share']<target_SD, 'trade_share'] = target_SD

    relative_sd_bounds = trade_with_demand.set_index(['from_abbreviation','to_abbreviation'])['trade_share'].squeeze().to_dict()
    
    return relative_sd_bounds

def get_liberalization_targets(lib_scenario):
    """
    Get self-sufficiency and single-dominant partner targets based on trade liberalization scenario.
    """
    if lib_scenario == 'low':
        target_SS=0.7 # self-sufficiency in future should be at least 70% of self-sufficiency in 2020
        target_SD=0.3 # not more than 30% of demand should be met through any single exporting partner if a link doesnt exist in 2020, else imports from partner should be less than current share)
    elif lib_scenario == 'medium':
        target_SS=0.5
        target_SD=0.5
    else:
        target_SS=0.3
        target_SD=0.7

    return target_SS, target_SD

def shock_trade_clearance(country_info, bilateral_info, eps_val, sigma_val, crop_code, calibration_output_path, model_output, year_select, SSP,
                          scen, factor_error, error, error_scale = 100, max_iter = 3000, input_folder='Input'):
    print(SSP, crop_code,year_select,'running', datetime.datetime.now(), max_iter) 
    logging.info(f"{SSP}, {crop_code}, {year_select}, running, {datetime.datetime.now()}, {max_iter}") 

    target_SS, target_SD = get_liberalization_targets(scen[3]) ### get the targets based on trade liberalization scenario

    ###### READ DATA #####
    #### read the calibration output #
    if year_select == 2020:
        ### base year ####
        trade_calib, prodprice_calib, conprice_calib, tc_calib, calib_constant =  read_calibration_output(calibration_output_path, crop_code)

    else: ### read input from previous calibration step    
        ### read IMPACT model output ##
        future_demand_elas = pd.read_csv(f'../../OPSIS/Data/Trade_clearance_model/{input_folder}/Future_scenarios/{SSP}/IMPACT_future_demand_elas.csv')
        future_supply_scaling = pd.read_csv(f'../../OPSIS/Data/Trade_clearance_model/{input_folder}/Future_scenarios/{SSP}/supply_scn/IMPACT_future_supply_{crop_code}.csv')
        future_demand_scaling = pd.read_csv(f'../../OPSIS/Data/Trade_clearance_model/{input_folder}/Future_scenarios/{SSP}/demand_scn/IMPACT_future_demand_{crop_code}.csv')

        ### Supply elas ##
        supply_elas = pd.DataFrame(country_info.supply_elas).reset_index()

        ## update data ##
        country_output, country_output_2020 = read_update_future_model_data(model_output, crop_code, year_select, SSP, scen, future_demand_elas, supply_elas, future_demand_scaling, future_supply_scaling)

        ### trade output ###
        trade_output = pd.read_csv(f'{model_output}Trade_output/trade_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{scen[3]}_{year_select-5}_{crop_code}.csv')
        trade_output['trade_binary'] = np.where(trade_output['trade']>0,1,0)
        trade_output['trade'] = np.round(np.where(trade_output['trade']<=error, error, trade_output['trade']), factor_error)

        ### self-sufficiency to domestic supply in 2020 ##
        country_output_2020['dom_share_demand'] = target_SS * country_output_2020['dom_share_demand']
        supply_balance = country_output_2020[['abbreviation','dom_share_demand']].set_index(['abbreviation']).squeeze().to_dict()
        # print(country_output_2020['dom_share_demand'].describe())

        # Calculate relative SD bounds based on historical trade shares
        relative_sd_bounds = calculate_historical_trade_shares(model_output, crop_code, SSP, scen, target_SD)

        ### set the values to initilize the model ##
        trade_calib, prodprice_calib, conprice_calib, tc_calib, calib_constant =  read_calibration_output_future(calibration_output_path, country_output, trade_output, crop_code, factor_error, error)
        print('input prod price')
        print(pd.DataFrame([prodprice_calib]).transpose().describe())
        print('input cons price')
        print(pd.DataFrame([conprice_calib]).transpose().describe())
        logging.info('input prod price')
        logging.info(pd.DataFrame([prodprice_calib]).transpose().describe())
        logging.info('input cons price')
        logging.info(pd.DataFrame([conprice_calib]).transpose().describe())

        ### update the country info, demand, supply and demand elas ###
        country_info = update_country_dict(country_dict=country_info, country_output=country_output, year_select=year_select)
        bilateral_info = update_bilateral_dict(bilateral_dict=bilateral_info, trade_output=trade_output)


    ####-------- INITIALIZE THE MODEL --------######
    model2 =  ConcreteModel()
    model2.i = Set(initialize=country_info.abbreviation, doc='Countries')

    ####-------- PARAMETERS --------######
    
    model2.prodprice03 = Param(model2.i, initialize= prodprice_calib, doc='production price 03')
    
    model2.conprice03 = Param(model2.i,initialize= conprice_calib,doc='consumer price 03')
    model2.tc03 = Param(model2.i, model2.i, initialize= tc_calib,doc='transportation cost 03')
    model2.calib = Param(model2.i, model2.i, initialize= calib_constant,doc='calibration cost 03')

    # ## trade calibration ##
    model2.trade_calib = Param(model2.i, model2.i, initialize= trade_calib,doc='trade_calib 03')

    if year_select != 2020: 
        model2.self_supply = Param(model2.i, initialize= supply_balance,doc='self-supply 03')
        model2.relative_sd_bound = Param(model2.i, model2.i, initialize=relative_sd_bounds, doc='relative supplier diversification bounds')

    ### tariffs
    model2.adv = Param(model2.i, model2.i, initialize=bilateral_info.adv.to_dict(),doc='tariff')

    #### Demand and Supply Elasticities ####
    model2.Ed = Param(model2.i,initialize=country_info.demand_elas.to_dict(),doc='demand elasticity') ### divided by 1,000
    model2.Es = Param(model2.i,initialize=country_info.supply_elas.to_dict(),doc='supply elasticity') ### divided by 1,000

    ### baseline demand and supply ###
    if year_select==2020:
        demand_2020, supply_2020 = get_calibrated_demand_supply_2020(crop_code, calibration_output_path, error, error_scale)
        model2.demand03 = Param(model2.i, initialize=demand_2020, doc='demand initial')
        model2.supply03 = Param(model2.i, initialize=supply_2020, doc='supply initial')
    else:
        model2.demand03 = Param(model2.i,initialize=(country_info.demand  +(error/error_scale)*len(country_info.demand)).to_dict(),doc='demand initial')
        model2.supply03 = Param(model2.i, initialize=(country_info.supply +(error/error_scale)*len(country_info.supply)).to_dict(),doc='supply initial')

    ### set parameters ##
    model2.epsilon = Param(initialize=0.001,doc='eps')
    model2.eps = Param(initialize=eps_val,doc='eps')
    model2.sigma = Param(initialize=sigma_val,doc='sigma')
    model2.existing_trade_binary = Param(model2.i, model2.i, initialize=bilateral_info.trade_binary.to_dict(),doc='binary existing trade')
    model2.existing_trade = Param(model2.i, model2.i, initialize = bilateral_info.trade_old.to_dict(),doc='existing trade')


    #### Create the demand and supply curves #####
    def B(model2, i):
        if model2.demand03[i]>1:
            return (model2.conprice03[i])/(model2.demand03[i]*model2.Ed[i])
        else:
            ### take average
            B_sum = 0; count = 0
            for j in model2.i:
                if model2.demand03[j]>1:
                    B_sum+=(model2.conprice03[j])/(model2.demand03[j]*model2.Ed[j])
                    count+=1

            return B_sum/count

    def D(model2, i):
        if model2.supply03[i]>1:
            return model2.prodprice03[i]/(model2.supply03[i]*model2.Es[i])
        else:
            ### take average
            D_sum = 0; count = 0
            for j in model2.i:
                if model2.supply03[j]>1:
                    D_sum+=model2.prodprice03[j]/(model2.supply03[j]*model2.Es[j])
                    count+=1

            return D_sum/count

    def A(model2, i):
        if model2.demand03[i]>1:
            return (model2.conprice03[i]) + model2.B[i] * model2.demand03[i]
        else:
            return (model2.conprice03[i]) + model2.B[i] * model2.demand03[i] - model2.epsilon

    def C(model2, i):
        if model2.supply03[i]>1:
            return model2.prodprice03[i] - model2.D[i] * model2.supply03[i]
        else:
            return model2.prodprice03[i] - model2.D[i] * model2.supply03[i] + model2.epsilon

    model2.B = Param(model2.i,initialize=B,doc='B: Absolute value of the inverse demand function slopes p=f(q)')
    model2.A = Param(model2.i,initialize=A,doc='A: Inverse demand function intercepts p=f(q)')

    model2.D = Param(model2.i,initialize=D,doc='D: Absolute value of the inverse supply function slopes p=f(q)')
    model2.C = Param(model2.i,initialize=C,doc='C: Inverse supply function intercepts p=f(q)')

    def tariff_initialize(model2, i, j):
        return (model2.prodprice03[i] + model2.tc03[i,j]-model2.calib[i,j]) * model2.adv[i,j]

    model2.tariff3 = Param(model2.i, model2.i, initialize = tariff_initialize,doc='tariff')

    ####-------- VARIABLES --------######
    model2.prodprice3 = Var(model2.i,initialize = model2.prodprice03.extract_values(), within = PositiveReals, doc='production price 3')
    model2.conprice3 = Var(model2.i,initialize = model2.conprice03.extract_values(),  within = PositiveReals, doc='consumer price 3')

    model2.demand = Var(model2.i,initialize=model2.demand03.extract_values(),  bounds = (len(country_info.demand)*(error/error_scale), None), doc='demand')
    model2.supply = Var(model2.i,initialize=model2.supply03.extract_values(),  bounds = (len(country_info.supply)*(error/error_scale), None), doc='supply')

    def trade_init(model2, i,j):
        if i == j:
            if model2.trade_calib[i,j] < model2.self_supply[j] * model2.demand03[j]:
                return error/error_scale+model2.self_supply[j] * model2.demand03[j]
            else:
                return model2.trade_calib[i,j]
        else:
            if model2.trade_calib[i,j] > model2.relative_sd_bound[i,j] * model2.demand03[j]:
                return model2.relative_sd_bound[i,j] * model2.demand03[j]
            else:
                return model2.trade_calib[i,j]
            
    def trade_bounds(model2, i,j):
        if i == j:
            return (error/error_scale+model2.self_supply[j] * model2.demand03[j], None)
        else:
            return (error/error_scale, model2.relative_sd_bound[i,j] * model2.demand03[j])
    
    
    if year_select == 2020:
        model2.trade3 = Var(model2.i, model2.i, initialize=trade_calib,  bounds=(error/error_scale, None), doc='trade3')
    else:
        model2.trade3 = Var(model2.i, model2.i, initialize=trade_init,  bounds=trade_bounds, doc='trade3')


    ####-------- EQUATIONS --------######
    def eq_PROD(model2, i):
            return complements(model2.supply[i] >= sum(model2.trade3[i,j] for j in model2.i), model2.prodprice3[i]>0)

    def eq_DEM(model2, i):
            return complements(sum(model2.trade3[j,i] for j in model2.i) >= model2.demand[i], model2.conprice3[i]>0)

    def eq_DPRICEDIF(model2, i):
            return complements(model2.conprice3[i] >= model2.A[i] - model2.B[i] * model2.demand[i], model2.demand[i]>=len(country_info.demand)*(error/error_scale))

    def eq_SPRICEDIF(model2, i):
            return complements(model2.C[i] + model2.D[i] * model2.supply[i] >= model2.prodprice3[i] , model2.supply[i]>=len(country_info.supply)*(error/error_scale))

    def eq_PRLINK2(model2, i, j):
        if model2.existing_trade_binary[i,j]==1:
            return complements(model2.prodprice3[i] + model2.calib[i,j]+ model2.tariff3[i,j] + (model2.tc03[i,j]-model2.calib[i,j]) * pow(model2.trade3[i,j]/model2.existing_trade[i,j], 1/model2.eps) >= model2.conprice3[j], model2.trade3[i,j]>=(error/error_scale))
        else:
            return complements(1.2* model2.prodprice3[i] + model2.calib[i,j] + model2.tariff3[i,j]+ (model2.tc03[i,j]-model2.calib[i,j]) + model2.sigma * model2.trade3[i,j]  >= model2.conprice3[j], model2.trade3[i,j]>=(error/error_scale))

    ### add constraints
    model2.eq_PROD = Complementarity(model2.i, rule = eq_PROD, doc='Supply >= quantity shipped')
    model2.eq_DEM = Complementarity(model2.i, rule = eq_DEM, doc='Demand <= quantity shipped')
    model2.eq_DPRICEDIF = Complementarity(model2.i, rule = eq_DPRICEDIF, doc='difference market demand price and local demand price')
    model2.eq_SPRICEDIF = Complementarity(model2.i, rule = eq_SPRICEDIF, doc='difference market supply price and local supply price')
    model2.eq_PRLINK2 = Complementarity(model2.i, model2.i, rule = eq_PRLINK2, doc='price chain 2')

    ####-------- SOLVE --------######
    TransformationFactory('mpec.simple_nonlinear').apply_to(model2)

    ### choose solver
    opt = SolverFactory('ipopt', solver_io='nl')
    opt.options['linear_solver'] = 'ma27'
    opt.options['nlp_scaling_method'] = 'user-scaling'
    opt.options['tol'] = 0.1
    opt.options['acceptable_tol'] = 0.1
    opt.options['max_iter'] = max_iter
    opt.options['max_cpu_time'] = 60 * 20 ### 20 min##
    # opt.options['hsllib'] = 'libcoinhsl.dylib'


    result=opt.solve(model2)

    initial_supply = np.sum(list(model2.supply03.extract_values().values()))/1e6
    initial_demand = np.sum(list(model2.demand03.extract_values().values()))/1e6
    output_supply =  np.sum(list(model2.supply.extract_values().values()))/1e6
    output_demand =  np.sum(list(model2.demand.extract_values().values()))/1e6
    print(year_select, initial_supply, output_supply, initial_demand, output_demand)
    logging.info(f"{year_select}, {initial_supply}, {output_supply}, {initial_demand}, {output_demand}")
    
    ### process output
    trade, country_output = process_final_output(model = model2, year_select= year_select, crop_code = crop_code, SSP = SSP, scen = scen, error = error, error_scale = error_scale)

    return trade, country_output
#
