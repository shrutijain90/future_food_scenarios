# Usage: python -m future_trade.spatial_trade_model.Trade_clearance_model_future

import sys
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

from future_trade.spatial_trade_model.functions_general import *
from future_trade.spatial_trade_model.functions_calibration import *
from future_trade.spatial_trade_model.functions_future import *
from pathos.multiprocessing import ProcessPool, cpu_count

###### additional information,  ####
sigma_val = 10 ### base = 10,
eps_val = 3 ### base = 7
epsilon = 0.001

factor_error = 3 ### base = 3
error = 1*10**(-1*factor_error)
error_scale = 100 ### base = 10

### paths for calibration input and output
data_dir = '../../OPSIS/Data/Trade_clearance_model'
calibration_output = f'{data_dir}/Output/Calibration/'
model_output = f'{data_dir}/Output/Trade_allocation_future/'
input_folder = 'Input'

### settings ###
max_iter = 50000 
SSP = 'SSP2'

### Scenarios ###
scen_diet = ['BMK', 'FLX', 'PSC', 'VEG', 'VGN'] 
scen_cal = ['2500kcal'] 
scen_clim = ['2.6', '7'] 
scen_lib = ['low', 'med', 'high'] 

# full diets: both RCPs, all three trade scenarios
scen_list = list(itertools.product(*[scen_diet, scen_cal, scen_clim, scen_lib]))
# 50% adoption diets: RCP 2.6, low and med trade scenario only
scen_list += list(itertools.product(*[['FLX50', 'PSC50', 'VEG50', 'VGN50'], scen_cal, ['2.6'], ['low', 'med']]))

crops_to_run = [
        'jwhea', 'jcass', 'jpigp', 'jgrnd',   
        'jrice', 'jpota', 'jlent', 'jothr',   
        'jmaiz', 'jyams', 'jchkp', 'jrpsd',   
        'jvege', 'jbarl', 'jswpt', 'jsnfl',   
        'jbean', 'jmill', 'jorat', 'jtols',   
        'jsoyb', 'jsorg', 'jbana', 'jpalm',   
        'jsubf', 'jocer', 'jplnt', 'jsugb',   
        'jtemf', 'jcowp', 'jopul', 'jsugc',   
        ]

### one log file per run, named after the first crop in that run ###
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
run_log = open(f"{model_output}run_log_{crops_to_run[0]}.txt", 'a')
sys.stdout = Tee(sys.__stdout__, run_log)
sys.stderr = Tee(sys.__stderr__, run_log)


#### run models ###
for crop_code in crops_to_run: 
    file_country = f'{data_dir}/{input_folder}/Country_data/country_information_{crop_code}.csv'
    file_bil = f'{data_dir}/{input_folder}/Trade_cost/bilateral_trade_cost_{crop_code}.csv'

    if crop_code in ['jyams', 'jchkp', 'jcowp', 'jlent', 'jpigp', 'jopul', 'jothr']: 
        error_scale = 1000
        epsilon = 0.0001
    else:
        error_scale = 100
        epsilon = 0.001
    
    for scen in scen_list:

        print(scen)
        
        if crop_code in ['jgrnd',  'jothr', 'jrpsd', 'jsnfl', 'jtols', 'jpalm', 'jsugc', 'jsugb'] and scen[0] not in ['BMK', 'FLX', 'FLX50']:
            continue

        country_output_all, trade_all = pd.DataFrame(), pd.DataFrame()
        for year_select in [2020, 2025, 2030, 2035, 2040, 2045, 2050]: 
            print(year_select)
            ### read country data ###
            country_class, bilateral_class = read_model_input(crop_code, factor_error, file_country, file_bil)

            ### run in parallel
            trade, country_output = shock_trade_clearance(country_class,
                                                        bilateral_class,
                                                        eps_val,
                                                        sigma_val,
                                                        epsilon,
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
                                                        input_folder) ###

            ## output
            country_output.to_csv(f'{model_output}Country_output/country_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{scen[3]}_{year_select}_{crop_code}.csv', index=False)
            trade.to_csv(f'{model_output}Trade_output/trade_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{scen[3]}_{year_select}_{crop_code}.csv', index=False)

            ## concat ##
            country_output_all = pd.concat([country_output_all, country_output])
            trade_all = pd.concat([trade_all, trade])

            print(datetime.datetime.now())

        # for year_select in [2020, 2025, 2030, 2035, 2040, 2045, 2050]:
        #     country_output_all = pd.concat([country_output_all, pd.read_csv(f'{model_output}Country_output/country_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{scen[3]}_{year_select}_{crop_code}.csv')])
        #     trade_all = pd.concat([trade_all, pd.read_csv(f'{model_output}Trade_output/trade_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{scen[3]}_{year_select}_{crop_code}.csv')])
        
        country_output_all.to_csv(f'{model_output}Country_output/country_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{scen[3]}_{crop_code}.csv', index=False)
        trade_all.to_csv(f'{model_output}Trade_output/trade_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{scen[3]}_{crop_code}.csv', index=False)
