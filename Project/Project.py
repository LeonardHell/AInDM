#python -m pip install pandas
from pathlib import Path

import pandas as pd

#data = pd.read_csv("/eu_annual_generation_mix.csv")
Energy_sources = ["Coal", "Natural Gas", "Hydro", "Nuclear", "Oil", "Other", "Solar", "Wind"]
data_prices = pd.read_csv(
    Path(__file__).resolve().parent
    / "data"
    / "Fraunhofer_ISE_LCOE_2024_2035_2045.csv"
)
print(data_prices)