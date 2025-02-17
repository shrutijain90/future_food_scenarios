"""
Author: Jasper Verschuur
Edited by: Shruti Jain
"""

import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.mpec import *
import math
import datetime
from future_trade.spatial_trade_model.functions_general import *

data_dir = '../../OPSIS/Data/Trade_clearance_model'

###-------------------------############# STEP 1 CALIBRATION ######## --------------------------------------####

### OBJECTIVE ####
def transport_cost_model(country_info, bilateral_info, sigma_val, eps_val, error, linear = 'no'):
    factor_error = int(np.abs(np.log10(error)))
    ######### INITIALIZE MODEL #########
    model = ConcreteModel()

    ######## SET COUNTRIES #########
    model.i = Set(initialize = country_info.abbreviation, doc='Countries abbreviation codes')

    ####### PARAMETERS ######
    ### parameters sigma and eps  ##
    model.sigma = Param(initialize = sigma_val, doc='sigma') ### linear so x1,000 tonnes
    model.eps = Param(initialize = eps_val, doc='eps') ### relative so scale independent

    ## observed trade ##
    model.trade01 = Param(model.i, model.i, initialize=bilateral_info.trade01.to_dict(), doc='initial trade')
    ### trade cost ##
    model.tc1 = Param(model.i, model.i, initialize= bilateral_info.tc1.to_dict(), doc='initial transport')
    ### tariff ##
    model.adv = Param(model.i, model.i, initialize= bilateral_info.adv.to_dict(), doc='tariff')
    ### production price ##
    model.prodprice1 = Param(model.i, initialize= country_info.production_cost.to_dict(), doc='initial producer price')

    ### existing trade and binary
    model.existing_trade_binary = Param(model.i, model.i, initialize = bilateral_info.trade_binary.to_dict(), doc='binary existing trade')
    model.existing_trade = Param(model.i, model.i, initialize = bilateral_info.trade_old.to_dict(), doc='existing trade')

    def tariff_eq1(model, i, j):
            return (model.prodprice1[i] + model.tc1[i,j]) * model.adv[i,j]

    ### tariff in USD /t ###
    model.tariff = Param(model.i, model.i, initialize = tariff_eq1, doc='tariff')

    def excess_demand_eq1(model, i):
        return sum(model.trade01[j,i] - model.trade01[i,j] for j in model.i)

    ### excess demand ###
    model.e = Param(model.i, initialize = excess_demand_eq1, doc='excess demand')

    ######## VARIABLES ######
    model.trade1 = Var(model.i, model.i, initialize = bilateral_info.trade_old.to_dict(), bounds = (error, None), doc='Trade1')

    ######## CONSTRAINTS #######
    def balance_eq1(model, i):
        return (model.e[i] == sum(model.trade1[j,i] - model.trade1[i,j] for j in model.i))

    def internal_trade_eq1(model, i):
        return (model.trade1[i,i] == model.trade01[i,i])

    model.balance = Constraint(model.i, rule=balance_eq1, doc='balance')
    model.internal_trade = Constraint(model.i, rule = internal_trade_eq1, doc='internal trade')

    ######## OBJECTIVE ########
    def objective_rule_linear_eq1(model):
        TTC = 0
        for i in model.i:
            for j in model.i:
                if model.existing_trade_binary[i,j]==1:
                    TTC += (model.tariff[i,j] + model.tc1[i,j]) * model.trade1[i,j]
                else:
                    TTC += (0.2* model.prodprice1[i] + model.tariff[i,j] + model.tc1[i,j]) * model.trade1[i,j]
        return TTC

    def objective_rule_non_linear_eq1(model):
        TTC = 0
        for i in model.i:
            for j in model.i:
                if model.existing_trade_binary[i,j]==1:
                    TTC += model.tariff[i,j] * model.trade1[i,j] + model.eps/(1+model.eps) * model.tc1[i,j] * model.trade1[i,j] * pow(model.trade1[i,j]/model.existing_trade[i,j], (1+model.eps)/model.eps)
                else:
                    TTC += (0.2* model.prodprice1[i] + model.tariff[i,j] + model.tc1[i,j]) * model.trade1[i,j] + 0.5 * model.sigma * pow(model.trade1[i,j], 2)
        return TTC


    if linear == 'yes':
        model.objective = Objective(rule=objective_rule_linear_eq1, sense=minimize, doc='Define objective function')
        ## Display of the output ##
        results = SolverFactory("glpk").solve(model)
    else:
        model.objective = Objective(rule=objective_rule_non_linear_eq1, sense=minimize, doc='Define objective function')
        ## Display of the output ##
        opt = SolverFactory("ipopt", solver_io='nl')
        opt.options['tol'] = 1e-8

        results = opt.solve(model)


    #### output trade calibration ####
    trade_calibration = np.round(pd.Series(model.trade1.extract_values()), factor_error)
    trade_calibration.mask(trade_calibration <error, error, inplace=True)

    return model, trade_calibration



###-------------------------############# STEP 2 CALIBRATION ######## --------------------------------------####

def update_transport_cost(model):
    ## get the tc_val
    tc_val = dataframe_model_output(var1 = model.tc2, var2 = model.tc1)
    tc_val['dummy'] = np.where(tc_val['new']>tc_val['original'], 1,0)
    tc_val['calib'] = tc_val['dummy'] * (tc_val['new']-tc_val['original'])
    tc_val['calib'] = np.where(tc_val['calib']<=0, 0, tc_val['calib'])

    return tc_val


def output_calibration_file(calibration_model, output_file, crop_code, error):
    factor_error = int(np.abs(np.log10(error)))
    ### trade
    trade_calibration_2 = np.round(pd.Series(calibration_model.trade2.extract_values()), factor_error)
    trade_calibration_2.mask(trade_calibration_2 <error, error, inplace=True)
    trade_calibration_2.to_csv(output_file+'trade_calibration_'+crop_code+'.csv', header=False)

    ### production and consumption price
    pd.Series(calibration_model.prodprice2.extract_values()).to_csv(output_file+'prodprice_calibration_'+crop_code+'.csv', header=False)
    pd.Series(calibration_model.conprice2.extract_values()).to_csv(output_file+'conprice_calibration_'+crop_code+'.csv', header=False)

    ### tc2 and calib
    pd.Series(calibration_model.tc2.extract_values()).to_csv(output_file+'tc_calibration_'+crop_code+'.csv', header=False)
    pd.Series(calibration_model.calib.extract_values()).to_csv(output_file+'calib_calibration_'+crop_code+'.csv', header=False)


def trade_clearance_calibration(country_info, bilateral_info, sigma_val, eps_val, error, trade_calibration_step1, crop_code, output_file,  count_max = 30, mu_val = 0.01, wtc = 1, wp = 5, wx = 200, max_iter = 500):
    ### scaling factors
    factor_trade = (1200/(bilateral_info.trade01.sum()/1e3))**2
    factor_tc = (5.4/(bilateral_info.tc1.sum()/1e6))**2
    factor_price = (100/(country_info.production_cost.sum()/1e3))**2

    #### basic settings
    count=0; count_max
    mu_val = mu_val
    d_mu = mu_val/2
    pen_val = 1e8

    print(mu_val, factor_trade, factor_tc, factor_price)

    ### start model
    model1 =  ConcreteModel()
    model1.i = Set(initialize=country_info.abbreviation, doc='Countries')

    #===================== PARAMETERS=============================================
    #### weighting factors ##
    model1.wtc = Param(initialize= (wtc*factor_tc)/len(country_info.abbreviation), mutable=True, doc='weight transport cost') ## base = wtc = 1
    model1.wp = Param(initialize= (wp*factor_price)/len(country_info.abbreviation), mutable=True,doc='weight market supply price') ## wp = 5
    model1.wx = Param(initialize= (wx *factor_trade)/(len(country_info.abbreviation)**2) , mutable=True,doc='weight trade') ### trade = 200 ###### WHY????

    ### slackness condition, = mutable
    model1.mu = Param(initialize=mu_val, mutable=True, doc='Parameter for the complementary slackness condition')

    ### sigma and eps
    model1.eps = Param(initialize=eps_val, doc='eps')
    model1.sigma = Param(initialize=sigma_val, doc='sigma')

    ### set trade, production cost, transport and tariff
    model1.trade01 = Param(model1.i, model1.i, initialize = bilateral_info.trade01.to_dict(), doc='initial trade')
    model1.prodprice1 = Param(model1.i, initialize = country_info.production_cost.to_dict(), doc='production price 1')
    model1.tc1 = Param(model1.i, model1.i, initialize = bilateral_info.tc1.to_dict(), doc='transportation cost 1')
    model1.adv = Param(model1.i, model1.i, initialize= bilateral_info.adv.to_dict(), doc='tariff')

    ### trade from model step 1
    model1.trade02 = Param(model1.i, model1.i, initialize =  trade_calibration_step1.to_dict(), doc='trade02, output from step1')

    ### existing trade ###
    model1.existing_trade_binary = Param(model1.i, model1.i, initialize= bilateral_info.trade_binary.to_dict(), doc='binary existing trade')
    model1.existing_trade = Param(model1.i, model1.i, initialize= bilateral_info.trade_old.to_dict(), doc='existing trade')

    ### tariff
    def tariff0_eq2(model1, i, j):
        return (model1.prodprice1[i] + model1.tc1[i,j]) * model1.adv[i,j]

    ### consumer price
    def consprice_eq2(model1, i):
        if model1.existing_trade_binary[i,i]==1:
            return model1.prodprice1[i] + model1.tc1[i,i]
        elif sum(model1.existing_trade_binary[j,i] for j in model1.i) > 0:
            min_val_list = []
            for j in model1.i:
                if model1.existing_trade_binary[j,i]==1:
                    min_val_list.append(model1.prodprice1[j] + model1.tc1[j,i] + model1.tariff02[j,i])
                else:
                    continue
            return min(min_val_list)
        else:
            return model1.prodprice1[i] + model1.tc1[i,i]


    ### tariff and consumer price functions ###
    model1.tariff02 = Param(model1.i, model1.i, initialize=tariff0_eq2, doc='tariff')
    model1.conprice02 = Param(model1.i, initialize= consprice_eq2, doc='consumer price 02', within = NonNegativeReals)


    #=====================  INITIAL ADJUSTMENT OF WEIGHTS =============================================

    def pi_startingpoint(model1, i, j):
            if model1.existing_trade_binary[i,j]==1:
                return np.abs(model1.tc1[i,j] * pow(model1.trade02[i,j]/model1.existing_trade[i,j], 1/model1.eps) + model1.tariff02[i,j] + model1.prodprice1[i] - model1.conprice02[j])
            else:
                return np.abs(1.2* model1.prodprice1[i] + model1.tc1[i,j] + model1.tariff02[i,j] + model1.sigma * model1.trade02[i,j] - model1.conprice02[j])

    def pen0_startingpoint(model1):
        return model1.mu * sum(model1.pi_startingpoint[i,j] * model1.trade02[i,j] for i in model1.i for j in model1.i)

    def z0_startingpoint(model1):
        return (model1.wtc  * sum(pow(model1.tc1[i,j] - model1.tc1[i,j], 2) for i in model1.i for j in model1.i)) + \
             (model1.wp  * sum(pow(model1.prodprice1[i] - model1.prodprice1[i], 2) for i in model1.i)) + \
            (model1.wx  * sum(pow(model1.trade02[i,j] - model1.trade01[i,j], 2) for i in model1.i for j in model1.i))

    ### find initial guess and update weights based on that
    model1.pi_startingpoint= Param(model1.i, model1.i, initialize=pi_startingpoint, doc='pi_startingpoint',  within = NonNegativeReals)
    model1.pen_startingpoint= Param(initialize=pen0_startingpoint, doc='pen_startingpoint', within = NonNegativeReals)
    model1.z_startingpoint= Param(initialize=z0_startingpoint, doc='z_startingpoint', within = NonNegativeReals)

    print(model1.z_startingpoint.extract_values()[None], model1.pen_startingpoint.extract_values()[None])

    scaling_weight = 0.01 *  model1.pen_startingpoint.extract_values()[None]/model1.z_startingpoint.extract_values()[None]

    model1.wtc = (scaling_weight) * (wtc*factor_tc)/len(country_info.abbreviation)
    model1.wp = (scaling_weight) * (wp*factor_price)/(len(country_info.abbreviation))
    model1.wx = (scaling_weight) *  (wx *factor_trade)/(len(country_info.abbreviation)**2)


    #=====================  INITIAL GUESS AT ERRORS  BASED ON UPDATED WEIGHTS =============================================
    def pi0_eq2(model1, i, j):
        if model1.existing_trade_binary[i,j]==1:
            return np.abs(model1.tc1[i,j] * pow(model1.trade02[i,j]/model1.existing_trade[i,j], 1/model1.eps) + model1.tariff02[i,j] + model1.prodprice1[i] - model1.conprice02[j])
        else:
            return np.abs(1.2* model1.prodprice1[i] + model1.tc1[i,j] + model1.tariff02[i,j] + model1.sigma * model1.trade02[i,j] - model1.conprice02[j])

    def pen0_eq2(model1):
        return model1.mu * sum(model1.pi0[i,j] * model1.trade02[i,j] for i in model1.i for j in model1.i)

    def z0_eq2(model1):
        return (model1.wtc  * sum(pow(model1.tc1[i,j] - model1.tc1[i,j], 2) for i in model1.i for j in model1.i)) + \
             (model1.wp  * sum(pow(model1.prodprice1[i] - model1.prodprice1[i], 2) for i in model1.i)) + \
            (model1.wx  * sum(pow(model1.trade02[i,j] - model1.trade01[i,j], 2) for i in model1.i for j in model1.i))

    def zz0_eq2(model1):
        return model1.z0 + model1.pen0

    model1.pi0= Param(model1.i, model1.i, initialize=pi0_eq2, doc='pi0', within = NonNegativeReals)
    model1.pen0= Param(initialize=pen0_eq2, doc='pen0', within = NonNegativeReals)
    model1.z0= Param(initialize=z0_eq2, doc='z0', within = NonNegativeReals)
    model1.zz0= Param(initialize=zz0_eq2, doc='zz0', within = NonNegativeReals)

    print(model1.z0.extract_values()[None], model1.pen0.extract_values()[None])

    ### calibration constant ###
    model1.calib= Param(model1.i, model1.i, mutable=True, initialize = 0,  doc='calib')

    #=====================  VARIABLES TO ESTIMATE =============================================
    model1.z= Var(initialize=model1.z0.extract_values()[None], within = NonNegativeReals, doc='z')
    model1.zz= Var(initialize=model1.zz0.extract_values()[None], within = NonNegativeReals, doc='zz')

    def tc2_bounds_eq2(model1, i,j):
        return (0.7 * model1.tc1[i,j], 3.0 * model1.tc1[i,j])

    model1.tc2= Var(model1.i, model1.i, initialize=model1.tc1.extract_values(), bounds = tc2_bounds_eq2,  doc='tc2')

    def prodprice_bounds_eq2(model1, i):
        return (0.7 * model1.prodprice1[i], 2 * model1.prodprice1[i])

    model1.prodprice2= Var(model1.i, initialize=model1.prodprice1.extract_values(), bounds = prodprice_bounds_eq2, doc='prodprice2')
    model1.conprice2= Var(model1.i, initialize=model1.conprice02.extract_values(), within = NonNegativeReals, doc='conprice2')


    model1.trade2= Var(model1.i, model1.i, initialize=model1.trade02.extract_values(), bounds = (error, None), doc='trade2')

    model1.pi= Var(model1.i, model1.i, initialize=model1.pi0.extract_values(), within = NonNegativeReals, doc='pi')
    model1.pen= Var(initialize=model1.pen0.extract_values()[None], within = NonNegativeReals, doc='pen')

    model1.tariff2 = Var(model1.i, model1.i, initialize=tariff0_eq2, doc='tariff')

    def tariff_eq2(model1, i, j):
        return model1.tariff2[i,j] == (model1.prodprice2[i] + (model1.tc2[i,j] - model1.calib[i,j])) * model1.adv[i,j]

    model1.tariff_constraint = Constraint(model1.i, model1.i, rule = tariff_eq2, doc='tariff constraint')


    #=====================  EQUATIONS =============================================
    def obj_eq2(model1):
        return model1.z == \
            (model1.wtc  * sum(pow(model1.tc2[i,j] - model1.tc1[i,j], 2) for i in model1.i for j in model1.i)) + \
            (model1.wp  * sum(pow(model1.prodprice2[i] - model1.prodprice1[i], 2) for i in model1.i)) + \
            (model1.wx  * sum(pow(model1.trade2[i,j] - model1.trade01[i,j], 2) for i in model1.i for j in model1.i))

    def PRLINK1_eq2(model1, i, j):
        if model1.existing_trade_binary[i,j]==1:
             return model1.prodprice2[i] + model1.calib[i,j] + model1.tariff2[i,j] + (model1.tc2[i,j]-model1.calib[i,j]) * pow(model1.trade2[i,j]/model1.existing_trade[i,j], 1/model1.eps) == model1.conprice2[j] + model1.pi[i,j]
        else:
            return 1.2* model1.prodprice2[i] + model1.calib[i,j] + (model1.tc2[i,j]-model1.calib[i,j]) + model1.tariff2[i,j] + model1.sigma * model1.trade2[i,j]  == model1.conprice2[j] + model1.pi[i,j]

    def pen_eq2(model1):
        return model1.mu * sum(model1.pi[i,j] * model1.trade2[i,j] for i in model1.i for j in model1.i) == model1.pen

    model1.obj = Constraint(rule = obj_eq2, doc='eq error')
    model1.eq_PRLINK1 = Constraint(model1.i, model1.i, rule = PRLINK1_eq2, doc='eq price link')
    model1.eq_pen = Constraint(rule = pen_eq2, doc='eq penalty')


    #===================== OBJECTIVE  =============================================
    def zz_eq2(model1):
        return model1.z + model1.pen

    model1.objective = Objective(rule = zz_eq2, sense = minimize, doc='Define objective function')

    #===================== SOLVE  =============================================
    #### settings for solver ###
    opt = SolverFactory("ipopt", solver_io='nl')
    opt.options['nlp_scaling_method'] = 'user-scaling'
    opt.options['halt_on_ampl_error'] = 'yes'
    opt.options['tol'] = 1e-3
    opt.options['mumps_mem_percent'] = 10000
    opt.options['max_iter'] = max_iter

    ## solve ##
    results = opt.solve(model1,tee = True)
    #results = opt.solve(model1)

    ### print initial error ###
    trade_error = np.sum((pd.Series(model1.trade2.extract_values())-pd.Series(model1.trade01.extract_values()))**2)/1e6
    transport_error = np.sum((pd.Series(model1.tc2.extract_values())-pd.Series(model1.tc1.extract_values()))**2)/1e6
    price_error = np.sum((pd.Series(model1.prodprice2.extract_values())-pd.Series(model1.prodprice1.extract_values()))**2)/1e6

    ### get pen value ##
    pen_val, z_val = model1.pen.get_values()[None], model1.z.get_values()[None]
    print(count, np.round(mu_val, 4), z_val, pen_val, pen_val/mu_val, datetime.datetime.now())
    print('trade error:',trade_error, trade_error*model1.wx.extract_values()[None])
    print('transport error:',transport_error, transport_error*model1.wtc.extract_values()[None])
    print('price error:',price_error, price_error*model1.wp.extract_values()[None])

    #===================== LOOP AND SOLVE AGAIN  =============================================

    while (pen_val/mu_val >= 1) & (count<count_max):
        #### add penality and update model
        mu_val = mu_val+ d_mu ### update mu_val
        model1.mu = mu_val

        ### add count
        count +=1
        ### update transport cost and calib
        tc_val = update_transport_cost(model = model1)

        ### update calibration
        model1.calib.clear()
        model1.calib._constructed = False
        model1.calib.construct(tc_val.set_index(['from_abbreviation','to_abbreviation'])['calib'].to_dict())

        #===================== CHECK IF IT SOLVES, IF NOT BREAK  =============================================
        try:
            ### solve again
            results = opt.solve(model1)

            ### errors
            pen_val, z_val = model1.pen.get_values()[None], model1.z.get_values()[None]

            trade_error = np.sum((pd.Series(model1.trade2.extract_values())-pd.Series(model1.trade01.extract_values()))**2)/1e6
            transport_error = np.sum((pd.Series(model1.tc2.extract_values())-pd.Series(model1.tc1.extract_values()))**2)/1e6
            price_error = np.sum((pd.Series(model1.prodprice2.extract_values())-pd.Series(model1.prodprice1.extract_values()))**2)/1e6

            print(count, np.round(mu_val, 4), z_val, pen_val, pen_val/mu_val, datetime.datetime.now())
            print('trade error:',trade_error, trade_error*model1.wx.extract_values()[None])
            print('transport error:',transport_error, transport_error*model1.wtc.extract_values()[None])
            print('price error:',price_error, price_error*model1.wp.extract_values()[None])
            final_update = 0

        except:
            final_update = 1
            print(count, np.round(mu_val, 4), 'error')
            break

    if final_update == 0:
        ### update calibration
        tc_val = update_transport_cost(model = model1)

        model1.calib.clear()
        model1.calib._constructed = False
        model1.calib.construct(tc_val.set_index(['from_abbreviation','to_abbreviation'])['calib'].to_dict())

    #===================== OUTPUT CALIBRATION FILES  =============================================
    #### output parameters from model calibration as csv files to read in ####
    output_calibration_file(model1, output_file, crop_code, error)

    return model1
