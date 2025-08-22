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
        # 'jwhea', 
        'jrice', 
        # 'jmaiz', 
        # 'jbarl', 
        # 'jmill', 
        # 'jsorg', 
        # 'jocer', 
        # 'jcass', 
        # 'jpota', 
        # 'jyams', 
        # 'jswpt', 
        # 'jorat', 
        # 'jvege', 
        # 'jbana', 
        # 'jplnt', 
        # 'jsubf', 
        # 'jtemf', 
        # 'jbean', 
        # 'jchkp', 
        # 'jcowp', 
        # 'jlent', 
        # 'jpigp', 
        # 'jopul', 
        # 'jsoyb',
        # 'jgrnd', 
        # 'jothr', 
        # 'jrpsd', 
        # 'jsnfl', 
        # 'jtols', 
        # 'jpalm', 
        # 'jsugb', 
        # 'jsugc'
        ]:
        print(crop_code)
        logging.info(crop_code)
        file_country = f'{data_dir}/Input/Country_data/country_information_{crop_code}.csv'
        file_bil = f'{data_dir}/Input/Trade_cost/bilateral_trade_cost_{crop_code}.csv'
    
        ### read and process model input ###
        country_class, bilateral_class = read_model_input(crop_code, factor_error)
    
    
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

        # if crop_code in ['jmill', 'jyams', 'jswpt', 'jorat', 'jcowp', 'jpigp']:
        #     wtc = 1
        #     wp = 1
        # else:
        #     wtc = 1
        #     wp = 1

        # if crop_code=='jyams':
        #     wx = 600
        # elif crop_code=='jgrnd':
        #     wx = 25
        # else:
        #     wx = 500

        # count_max = 50
        # if crop_code=='jgrnd':
        #     count_max = 8
        # else:
        #     count_max = 30
        
        # # maize
        # wtc = 1
        # wp = 1
        # wx = 500
        # count_max = 50
        # maz_iter = 3000
        # scale_factor = 1

        # rice
        wtc = 1
        wp = 1
        wx = 300
        count_max = 50
        max_iter = 3000
        scale_factor = 0.5

        # wheat
        wtc = 1
        wp = 1
        wx = 300
        count_max = 50
        max_iter = 3000
        scale_factor = 0.5
        
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
