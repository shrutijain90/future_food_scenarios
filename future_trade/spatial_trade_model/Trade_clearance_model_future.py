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

### paths for calibration input and
data_dir = '../../OPSIS/Data/Trade_clearance_model'
calibration_output = f'{data_dir}/Output/Calibration/'
model_output = f'{data_dir}/Output/Trade_allocation_future/'

#### run models ###
crop_code = 'jwhea' 
### settings# ###
max_iter = 5000
SSP = 'SSP2'

### Scenarios ###
scen_diet = ['BMK','FLX','FLX_hredmeat', 'FLX_hmilk', 'PSC', 'VEG', 'VGN']
scen_cal = ['2100kcal','2500kcal']
scen_clim = ['none'] #'rcp2p6', 'rcp4p6', 'rcp6p0', 'rcp8p5'

scen_list = list(itertools.product(*[scen_diet, scen_cal, scen_clim]))

for scen in scen_list:
    print(scen)
    country_output_all, trade_all = pd.DataFrame(), pd.DataFrame()
    for year_select in [2020, 2025, 2030, 2035, 2040, 2045, 2050]:
        print(year_select)
        ### read country data ###
        country_class, bilateral_class = read_model_input(crop_code, factor_error)

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
                                                      max_iter) ###

        ## output
        country_output.to_csv(f'{model_output}Country_output/country_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{year_select}_{crop_code}.csv', index=False)
        trade.to_csv(f'{model_output}Trade_output/trade_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{year_select}_{crop_code}.csv', index=False)

        ## concat ##
        country_output_all = pd.concat([country_output_all, country_output])
        trade_all = pd.concat([trade_all, trade])

        print(datetime.datetime.now())

    country_output_all.to_csv(f'{model_output}Country_output/country_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{crop_code}.csv', index=False)
    trade_all.to_csv(f'{model_output}Trade_output/trade_output_{SSP}_{scen[0]}_{scen[1]}_{scen[2]}_{crop_code}.csv', index=False)
