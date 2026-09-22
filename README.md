# Overview

Code for modelling how international food trade could evolve under future dietary scenarios. 

## Repository structure

```
future_trade/
├── data_inputs/                            # build model inputs from raw datasets
│   ├── align_regions.py                        # aligns country/region classifications across sources
│   ├── balance_trade.py                        # constructs balanced bilateral trade matrices from FAOSTAT
│   ├── producer_prices.py                      # gap fills producer price data from FAOSTAT 
│   ├── export_processing_factors.py            # processing factors from FAOSTAT SUA/FBS
│   ├── dietary_scenarios.ipynb                 # EAT-Lancet dietary scenarios
│   ├── feed_and_other.ipynb                    # feed and other-use demand components
│   ├── merge_data.py                           # merges inputs for the base-year calibration model
│   └── merge_data_future.py                    # merges inputs for future scenario runs
│
├── spatial_trade_model/                    # the trade clearance model
│   ├── functions_general.py                    # shared helpers/classes used by calibration & future runs
│   ├── functions_calibration.py                # model for base-year calibration
│   ├── functions_future.py                     # model for future scenario solves
│   ├── Trade_clearance_model_calibration.py    # calibrates the model to the base year
│   ├── calibration_checks.ipynb                # validates calibrated model against observed data
│   ├── Trade_clearance_model_future.py         # solves future trade scenarios
│   ├── solver_outputs.ipynb                    # summarises future scenario solver outputs
│   └── figures.R                               # R script for figures
```

## The model

All optimisation is written in Pyomo (https://www.pyomo.org/) with HSL's MA27 as (non-linear) solver:https://github.com/coin-or-tools/ThirdParty-HSL. Alternatively, one can run the code using the standard ipopt solver in Pyomo.

Software requirements:
- Python Python 3.12.9
- Tested on macOS Tahoe 26.1

Runtime:
- Calibration runtime around 1 hour per crop.
- Future model runtime is around 15 minutes per modelled crop/scenario/year.  

## Data

Input scripts read from external data directories (e.g. `../../data/` and `../../OPSIS/Data/`) and include:  
- FAOSTAT bulk downloads 
- Country classifications
- EAT-Lancet future demand data
- IMPACT model outputs

These are not tracked in version control and must be available locally to run.

## Dependencies

pandas, numpy, matplotlib, scikit-learn, statsmodels, geopandas, pyomo.
