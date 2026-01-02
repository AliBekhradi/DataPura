# Importing
from sqlalchemy import create_engine
from mapping import MappingManager
import pandas as pd
from loaders import DataLoader
from rc_ops import RowsAndColumns
from encoding import Encoders
from string_ops import stringsprocessor
from NAS import NAS
from insights import Insights

# Intilization
engine = create_engine("postgresql://admin:10293847@localhost:5432/datapura")
mm = MappingManager()
load= DataLoader()
rc = RowsAndColumns()
info = Insights()
nas = NAS()
strp = stringsprocessor()
enc = Encoders()

# Opening
t = load.ingest(file_path= "C:/Users/ali/Projects/DataPura/tesla_deliveries.csv", return_df= True)
df = load.read_from_sql("tesla_deliveries")
# info.distribution_rates(engine=engine, table= "tesla_deliveries", columns= "year")
# info.min_max_finder(engine=engine, table= "tesla_deliveries", columns="year")
# info.missing_rows(engine=engine, table="tesla_deliveries", column="charging_stations")
info.unique_items_list(engine=engine, table="tesla_deliveries", columns= "charging_stations")


# Insights
"""info.unique_items_list (df, columns= df.columns, count= True)
info.min_max_finder (df, columns = ["Year", "Month", "Production_Units", "Avg_Price_USD", "Battery_Capacity_kWh", "Range_km", "CO2_Saved_tons", "Charging_Stations"])
df = info.missing_rows (df, drop_threshold= 0.8, axis= "rows", inplace= False)


# String Ops
df = strp.convert_case(df, columns= {}, mode= {})
df = strp.batch_string_remover(df, columns={}, remove={})
df = strp.remove_irrelevant_characters(df, columns={})
df = strp.remove_whitespace(df, columns={})

# R&C
df = rc.columns_drop(df, columns={})
df = rc.rename_columns(engine=engine, table= {}, columns={}, rename_to={})
df = rc.value_remover(df, value={}, columns={}, mode={})
df = rc.dup_row_remover(df, columns={}, keep={})
df = rc.imputator(df, columns={}, mode={}, impute_zeros={})

# Normalization And Standardization (NAS)
df = nas.normalize(df, columns={}, mapping_return={}, mapping_format={}, save_scaler= True)
df = nas.normalize_categories(df, columns={}, mapping={}, threshold={})
df = nas.format_numbers(df, columns={}, decimal_places=2, drop_invalid=False)
df = nas.standardize_dates(df, columns={})

# Encoding
df = enc.OneHotEncoder(df, columns={}, mapping_return=True)
df = enc.FrequencyEncoder(df, columns={}, mapping_return=True)
df = enc.TargetEncoder(df, columns={}, target={}, mapping_return=True)"""