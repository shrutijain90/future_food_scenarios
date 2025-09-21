# Usage: python -m future_trade.spatial_trade_model.Trade_clearance_model_future

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.mpec import *
import math
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import itertools
import logging

from future_trade.spatial_trade_model.functions_general import *
from future_trade.spatial_trade_model.functions_calibration import *
from future_trade.spatial_trade_model.functions_future import *
from pathos.multiprocessing import ProcessPool, cpu_count

###### additional information,  ####
sigma_val = 10 ### base = 10,
eps_val = 7 ### base = 7

factor_error = 3 ### base = 3
error = 1*10**(-1*factor_error)
error_scale = 100 ### base = 10

### paths for calibration input and output
data_dir = '../../OPSIS/Data/Trade_clearance_model'
calibration_output = f'{data_dir}/Output/Calibration/'
model_output = f'{data_dir}/Output/Trade_allocation_future/'
input_folder = 'Grouped_input'
logging.basicConfig(filename=f"{model_output}model_info.txt", level=logging.INFO, format="%(message)s")

### settings ###
max_iter = 5000
SSP = 'SSP2'
target_SS=0.3 # self-sufficiency in future should be at least 30% of self-sufficiency in 2020
target_SD=0.3 # not more than 30% of demand should be met through any single exporting partner

### Scenarios ###
scen_diet = ['BMK','FLX','FLX_hredmeat', 'FLX_hmilk', 'PSC', 'VEG', 'VGN']
scen_cal = ['2500kcal'] #'2100kcal' # taking 2500 cal as default means that we are assuming that the present is closer to 2500kcal scenario, as will be the future
scen_clim = ['NoCC', '2.6', '7', '8.5'] #

scen_list = list(itertools.product(*[scen_diet, scen_cal, scen_clim]))

#### run models ###
for crop_code in [
        'jwhea', 'jrice', 'jmaiz', 'jbarl', 'jmill', 'jsorg', 
        'jocer', 'jcass', 'jpota', 'jyams', 'jswpt', 'jorat', 
        'jvege', 'jbana', 'jplnt', 'jsubf', 'jtemf', 'jbean', 
        'jchkp', 'jcowp', 'jlent', 'jpigp', 'jopul', 'jsoyb',
        'jgrnd', 'jothr', 'jrpsd', 'jsnfl', 'jtols', 'jpalm', 
        'jsugb', 'jsugc'
        ]: 
    file_country = f'{data_dir}/{input_folder}/Country_data/country_information_{crop_code}.csv'
    file_bil = f'{data_dir}/{input_folder}/Trade_cost/bilateral_trade_cost_{crop_code}.csv'
    logging.info(crop_code)

    for scen in scen_list:
        print(scen)
        logging.info(scen)
        country_output_all, trade_all = pd.DataFrame(), pd.DataFrame()
        for year_select in [2020, 2025, 2030, 2035, 2040, 2045, 2050]:
            print(year_select)
            logging.info(year_select)
            ### read country data ###
            country_class, bilateral_class = read_model_input(crop_code, factor_error, file_country, file_bil)

            ### run in parallel
            trade, country_output = shock_trade_clearance(country_class,
                                                        bilateral_class,
                                                        eps_val,
                                                        sigma_val,
                                                        crop_code,
                                                        calibration_output,
                                                        model_output,
                                                        year_select,
                                                        SSP,
                                                        scen,
                                                        factor_error,
                                                        error, ### error
                                                        error_scale, ### default = 100
                                                        max_iter,
                                                        input_folder,
                                                        target_SS,
                                                        target_SD) ###

            ## output
            country_output.to_csv(f'{model_output}Country_output/country_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{year_select}_{crop_code}.csv', index=False)
            trade.to_csv(f'{model_output}Trade_output/trade_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{year_select}_{crop_code}.csv', index=False)

            ## concat ##
            country_output_all = pd.concat([country_output_all, country_output])
            trade_all = pd.concat([trade_all, trade])

            print(datetime.datetime.now())
            logging.info(datetime.datetime.now())

        country_output_all.to_csv(f'{model_output}Country_output/country_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{crop_code}.csv', index=False)
        trade_all.to_csv(f'{model_output}Trade_output/trade_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{crop_code}.csv', index=False)
