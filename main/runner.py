"""Importing"""
from core import Preprocessor
from core import MappingManager
import os
import pandas as pd

"""Intilization"""
prep = Preprocessor()
mm = MappingManager()

"""Opening"""
try:
    data_path = os.path.join(os.getcwd(), "vehicles.csv")
    df = pd.read_csv(data_path, header = 0)
    if df.empty:
        raise ValueError("Dataset is empty. ")
except FileNotFoundError:
    print("Error: file not found. Make sure you're typing the right name?")
    exit()
except ValueError as e:
    print(e)
    exit()
    
"""Insights"""
df = prep.unique_items_list (df, columns= [], count= {})
df = prep.min_max_finder (df, columns = [])
df = prep.missing_rows (df, drop_threshold= {}, axis= {}, inplace= {})
prep.save_dataframe (df, output_path={}, file_format= {})

"""Basics"""
df = prep.value_remover(df, value= [], columns= [], mode= [], error_skip= [])
df = prep.convert_case(df , columns= [], mode= [], error_skip= [])
df = prep.remove_whitespace(df , columns= [], mode= [], error_skip= [])
df = prep.standardize_dates(df , columns= [], output_format= [], error_skip= [])
df = prep.format_numbers(df, columns= [], decimal_places= [], drop_invalid= [], error_skip= [])
df = prep.columns_rename (df, columns=[], rename_to= [], strict= {})
df = prep.columns_drop (df, columns=[])
df = prep.rows_sampling (df, n= {})
df = prep.normalize_categories(df, columns= [], mapping= [], threshold=80, error_skip= [])
df = prep.remove_irrelevant_characters(df, columns= [], error_skip= [])
df = prep.save_dataframe (df, output_path={}, file_format= {})
df = prep.dup_row_remover (df, columns=[], keep= {})

"""Imputation"""
df = prep.imputator (df, columns=[], mode= {})
prep.save_dataframe (df, output_path={}, file_format= {})

"""Normalization (Scaling)"""
df = prep.normalize(df, mapping_return={}, mapping_format= {}, save_scaler= {})
prep.save_dataframe (df, output_path={}, file_format= {})

"""Encoding"""
df = prep.FrequencyEncoder(df, columns=[], mapping_return= {})
df = prep.OneHotEncoder(df, columns=[], mapping_return= {})
df = prep.TargetEncoder(df, columns=[], target={}, mapping_return={})
prep.save_dataframe(df, output_path={}, file_format= {})

"""Final Save"""
prep.save_dataframe(df, output_path={}, file_format= {})