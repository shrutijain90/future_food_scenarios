"""
Author: Jasper Verschuur
Edited by: Shruti Jain
"""

# Usage: python -m future_trade.spatial_trade_model.Trade_clearance_model_calibration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.mpec import *
import math
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import logging

from future_trade.spatial_trade_model.functions_general import *
from future_trade.spatial_trade_model.functions_calibration import *
from pathos.multiprocessing import ProcessPool, cpu_count

###### additional information,  ####
sigma_val = 10 ### base = 10,
eps_val = 7 ### base = 7

factor_error = 3
error = 1*10**(-1*factor_error)
data_dir = '../../OPSIS/Data/Trade_clearance_model'
calibration_output = f'{data_dir}/Output/Calibration/'
logging.basicConfig(filename=f"{calibration_output}calibration_info.txt", level=logging.INFO, format="%(message)s")


if __name__ == '__main__':
    #### read data ###
    for crop_code in [
        # 'jwhea', 'jrice',  'jmaiz', 'jsoyb',
        # 'jbarl', 
        'jcass', 
        # 'jvege', 'jbana', 'jbean', 
        # 'jgrnd', 'jrpsd', 'jpalm', 'jsugc',
        # 'jmill', 'jsorg', 'jocer', 'jpota', 
        # 'jyams', 'jswpt', 'jorat', 'jplnt', 
        # 'jsubf', 'jtemf', jchkp', 'jcowp',
        # 'jlent', 'jpigp', 'jopul', 'jothr', 
        # 'jsnfl', 'jtols', 'jsugb'
        ]:

        print(crop_code)
        logging.info(crop_code)
        file_country = f'{data_dir}/Input/Country_data/country_information_{crop_code}.csv'
        file_bil = f'{data_dir}/Input/Trade_cost/bilateral_trade_cost_{crop_code}.csv'
    
        ### read and process model input ###
        country_class, bilateral_class = read_model_input(crop_code, factor_error, file_country, file_bil) 
    
        ### solve the first model step ###
        logging.info('Model step 1')
        print(datetime.datetime.now())
        model_step1, trade_calibration_1 = transport_cost_model(country_info=country_class,
                                                                bilateral_info=bilateral_class,
                                                                sigma_val=sigma_val,
                                                                eps_val=eps_val,
                                                                error=error,
                                                                linear='no')
    
        print(datetime.datetime.now())
    
        ### model validation after step 1 ###
        model_validation_s1, hit_ratio_s1, MSE_s1, R2_s1 = compare_trade_flows(model_predict=model_step1.trade1,
                                                                               model_base=model_step1.trade01,
                                                                               error=error)
        
        logging.info(f'hit ratio: {np.round(hit_ratio_s1, 2)}')
        logging.info(f'MSE: {np.round(MSE_s1, 2)}')
        logging.info(f'R-squared: {np.round(R2_s1, 2)}')
    
        ### plot of trade flows ##
        scatter_plot_trade(df_output=model_validation_s1, 
                           fname=f'{calibration_output}{crop_code}_s1')
    
    
        #### run step 2 calibration ####
        logging.info('Model step 2')
        
        if crop_code in ['jvege', 'jbean', 'jgrnd']:
            wtc = 1
            wp = 1
            wx = 10000
            count_max = 50
            max_iter = 3000
            scale_factor = 5
        elif crop_code in ['jwhea', 'jrice', 'jsoyb']:
            wtc = 1
            wp = 1
            wx = 10000
            count_max = 50
            max_iter = 3000
            scale_factor = 1
        elif crop_code in ['jmaiz', 'jcass']:
            wtc = 1
            wp = 1
            wx = 10000
            count_max = 50
            max_iter = 3000
            scale_factor = 5
        elif crop_code in ['jbarl']:
            wtc = 1
            wp = 1
            wx = 10000
            count_max = 50
            max_iter = 3000
            scale_factor = 2
        
        model_calibration = trade_clearance_calibration(country_info=country_class,
                                                        bilateral_info=bilateral_class,
                                                        sigma_val=sigma_val,
                                                        eps_val=eps_val,
                                                        error=error,
                                                        trade_calibration_step1=trade_calibration_1,
                                                        crop_code=crop_code,
                                                        output_file=calibration_output,
                                                        count_max=count_max,  ### default = 25
                                                        mu_val=0.01,  ### default = 0.01
                                                        wtc=wtc,  ### default = 10
                                                        wp=wp,    ### default = 5
                                                        wx=wx,  ### default = 200
                                                        max_iter=max_iter,
                                                        scale_factor=scale_factor)  ### default = 500
    
    
        ### model validation after step 3 ###
        model_validation_s2, hit_ratio_s2, MSE_s2, R2_s2 = compare_trade_flows(model_predict=model_calibration.trade2,
                                                                               model_base=model_calibration.trade01,
                                                                               error=error)

        logging.info(f'hit ratio: {np.round(hit_ratio_s2, 2)}')
        logging.info(f'MSE: {np.round(MSE_s2, 2)}')
        logging.info(f'R-squared: {np.round(R2_s2, 2)}\n')
        
        ### plot of trade flows ##
        scatter_plot_trade(df_output=model_validation_s2, 
                           fname=f'{calibration_output}{crop_code}_s2')
        
        # # if both demand and supply for a region are 0, it gets omitted from output files. adding all regions so the future code runs as expected
        # df_country = pd.read_csv(file_country)
        # df_bil = pd.read_csv(file_bil)
        
        # trade_cal = pd.read_csv(f'{calibration_output}trade_calibration_{crop_code}.csv', header=None)
        # trade_cal.columns = ['from_abbreviation', 'to_abbreviation', 'trade_cal']
        # trade_cal = trade_cal.merge(df_bil[['from_abbreviation', 'to_abbreviation']], how='right').fillna(0)
        # trade_cal.to_csv(f'{calibration_output}trade_calibration_{crop_code}.csv', index=False, header=False)

        # calib = pd.read_csv(f'{calibration_output}calib_calibration_{crop_code}.csv', header=None)
        # calib.columns = ['from_abbreviation', 'to_abbreviation', 'calib']
        # calib = calib.merge(df_bil[['from_abbreviation', 'to_abbreviation']], how='right').fillna(0)
        # calib.to_csv(f'{calibration_output}calib_calibration_{crop_code}.csv', index=False, header=False)

        # tc = pd.read_csv(f'{calibration_output}tc_calibration_{crop_code}.csv', header=None)
        # tc.columns = ['from_abbreviation', 'to_abbreviation', 'tc']
        # tc = tc.merge(df_bil[['from_abbreviation', 'to_abbreviation']], how='right').fillna(0)
        # tc.to_csv(f'{calibration_output}tc_calibration_{crop_code}.csv', index=False, header=False)

        # conprice = pd.read_csv(f'{calibration_output}conprice_calibration_{crop_code}.csv', header=None)
        # conprice.columns = ['abbreviation', 'conprice']
        # conprice = conprice.merge(df_country[['abbreviation']], how='right').fillna(0)
        # conprice.to_csv(f'{calibration_output}conprice_calibration_{crop_code}.csv', index=False, header=False)

        # prodprice = pd.read_csv(f'{calibration_output}prodprice_calibration_{crop_code}.csv', header=None)
        # prodprice.columns = ['abbreviation', 'prodprice']
        # prodprice = prodprice.merge(df_country[['abbreviation']], how='right').fillna(0)
        # prodprice.to_csv(f'{calibration_output}prodprice_calibration_{crop_code}.csv', index=False, header=False)
