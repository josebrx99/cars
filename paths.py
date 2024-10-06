import pandas as pd
import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')).replace("\\", "/")

src_car        = f"{root}/car/src"
pkl_car        = f"{root}/car/pkl"
models_pkl_car = f"{root}/car/models_pkl"
temp_car       = f"{root}/car/temp"
data_car       = f"{root}/car/data"
metrics_car    = f"{root}/car/metrics"

src_bike        = f"{root}/bike/src"
pkl_bike        = f"{root}/bike/pkl"
models_pkl_bike = f"{root}/bike/models_pkl"
temp_bike       = f"{root}/bike/temp"
data_bike       = f"{root}/bike/data"
metrics_bike    = f"{root}/bike/metrics"
