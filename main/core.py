import pandas as pd
import re
from tqdm import tqdm
from typing import Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import os
import json
import joblib
import openpyxl
from dateutil.parser import parse
from rapidfuzz import process, fuzz
import numpy as np

class MappingManager:
    
    def __init__(self, save_path="mappings"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def save_mapping(self, column_name, mapping, format: str = "joblib"):
        format = format.lower().strip()
        file_path = os.path.join(self.save_path, f"{column_name}_mapping.{format}")

        if format == "json":
            with open(file_path, "w") as f:
                json.dump(mapping, f)
            print(f"✅ Mapping saved: {file_path}")

        elif format == "joblib":
            file_path = os.path.join(self.save_path, f"{column_name}_mapping.pkl")
            joblib.dump(mapping, file_path)
            print(f"✅ Mapping saved: {file_path}")

        else:
            print("❌ Please enter a valid format ('joblib' or 'json').")

    def load_mapping(self, column_name, format: str = "joblib"):
        format = format.lower().strip()
        file_path = os.path.join(self.save_path, f"{column_name}_mapping.{format}")

        if format == "json":
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    return json.load(f)
            else:
                print(f"❌ Mapping '{column_name}' (JSON) not found in '{self.save_path}'.")

        elif format == "joblib":
            file_path = os.path.join(self.save_path, f"{column_name}_mapping.pkl")
            if os.path.exists(file_path):
                return joblib.load(file_path)
            else:
                print(f"❌ Mapping '{column_name}' (Joblib) not found in '{self.save_path}'.")

        else:
            print("❌ Please enter a valid format ('joblib' or 'json').")

    def apply_mapping(self, series, mapping):
        return series.map(mapping)


class Preprocessor:
    
    tqdm.pandas()
    
    def __init__(self):
        pass
    
    def rows_sampling(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        
        if not isinstance(n, int):
            raise TypeError("The number of rows should be an integer")
        
        if isinstance(df, str):
            df = pd.read_csv(df, index_col= False)
        
        return df.sample(n=n, random_state=42, ignore_index= True)

    def value_rows_remover(self, df: pd.DataFrame, value: int, columns: Union[str, list], error_skip: bool = False):
        
        if isinstance(columns, str):
            columns = [columns]
        
        if not isinstance(value, int):
            raise TypeError("The value should be an integer.")
            
        for col in columns:
            if col not in df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")

        value_removed_df = df.copy()
        
        # Replace the given value with NaN only in the specified columns
        value_removed_df[columns] = value_removed_df[columns].progress_apply(lambda x: x.replace(value, pd.NA))

        # Drop only NaN values in the specified columns, but keep the other data
        value_removed_df = value_removed_df.dropna(subset=columns)

        return value_removed_df

    def imputator(self, 
              df: pd.DataFrame, 
              columns: Union[list, str], 
              mode: Union[list, str], 
              error_skip: bool = False,
              impute_zeros: bool = False) -> pd.DataFrame:
        
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(mode, str):
            mode = [mode] * len(columns)
        elif len(mode) != len(columns):
            raise ValueError("Length of mode list must match the number of columns.")
        
        for col in columns:
            if col not in df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")
        
        imputed_df = df.copy()
        
        for col, m in zip(columns, mode):
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
            
            # Replace zeros with NaN if the switch is on
            if impute_zeros:
                imputed_df[col] = imputed_df[col].replace(0, np.nan)

            if m == "mean":
                imputed_df[col] = imputed_df[col].fillna(imputed_df[col].mean())
            elif m == "median":
                imputed_df[col] = imputed_df[col].fillna(imputed_df[col].median())
            elif m == "mode":
                imputed_df[col] = imputed_df[col].fillna(imputed_df[col].mode()[0])
            elif m == "interpolate":
                imputed_df[col] = imputed_df[col].interpolate()
            else:
                raise ValueError(f"Invalid mode '{m}' for column '{col}'. Choose from: mean, median, mode, interpolate.")
        
        return imputed_df
    
    def distribution_rates(self, 
            df: pd.DataFrame, 
            columns: Union[list, str],
            error_skip: bool = False) -> pd.DataFrame:
        
        if isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col not in df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")
    
        result = {}

        for col in columns:
            total = len(df)
            null_count = df[col].isna().sum()
            zero_count = (df[col] == 0).sum()
            safe_count = total - null_count - zero_count

            result[col] = {
                "Safe (count)": safe_count,
                "Safe (%)": round(safe_count / total * 100, 2),
                "Zero (count)": zero_count,
                "Zero (%)": round(zero_count / total * 100, 2),
                "Null (count)": null_count,
                "Null (%)": round(null_count / total * 100, 2),
            }

        return pd.DataFrame(result).T
    
    def batch_string_remover(self, df: pd.DataFrame, columns: Union[list,str], remove: Union[list,str], error_skip: bool = False) -> pd.DataFrame:
        """
        Removes specific substrings from specified columns in batches.
        
        Example:
            columns = ["number of cylinders", "top speed", "0-60"]
            remove = ["cylinders", "mph", "s"]
        
        The function will remove "cylinders" from the first column,
        "mph" from the second, and "s" from the third.
        """
        
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(remove, str):
            remove = [remove]
        
        if len(columns) != len(remove):
            raise ValueError("The 'columns' and 'remove' lists must have the same length.")

        for col in columns:
            if col not in df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")
        
        string_removed_df = df.copy()        
        
        for col, to_remove in zip(columns, remove):
            string_removed_df[col] = string_removed_df[col].str.replace(to_remove, "", regex=True, flags=re.IGNORECASE).str.strip()

        return string_removed_df

    def columns_rename(self, df: pd.DataFrame, columns: Union[list, str], rename_to: Union[list, str], error_skip: bool = True) -> pd.DataFrame:
        
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(rename_to, str):
            rename_to = [rename_to]

        if len(columns) != len(rename_to):
            raise ValueError("Length of 'columns' and 'rename_to' must be the same.")

        missing = [col for col in columns if col not in df.columns]

        if missing and not error_skip:
            raise KeyError(f"Column(s) {missing} not found in DataFrame.")

        # If not strict, filter out missing ones before zip
        if error_skip:
            columns, rename_to = zip(*[
                (col, new) for col, new in zip(columns, rename_to)
                if col in df.columns
            ])
        renamed_df = df.copy()
        
        rename_map = dict(zip(columns, rename_to))
        renamed_df = renamed_df.rename(columns=rename_map)

        return renamed_df
    
    def columns_drop(self, df: pd.DataFrame, columns: Union[list, str], error_skip: bool = False) -> pd.DataFrame:
        
        if isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col not in df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")
        
        drop_df = df.copy()
        # Drop specified columns
        drop_df = drop_df.drop(columns=columns, axis=1)
        
        return drop_df
    
    def value_remover(self, df: pd.DataFrame, value: Union[int, list], columns: Union[str, list], mode: Union[str, list], error_skip: bool = False) -> pd.DataFrame:
        
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(value, int) or isinstance(value, tuple):
            value = [value]
        if isinstance(mode, str):
            mode = [mode]
        
        if not (len(columns) == len(value) == len(mode)):
            raise ValueError("Columns, values, and modes must have the same length.")
        
        for col in columns:
            if col not in df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")
        
        value_removed_df = df.copy()
        
        for col, val, mod in zip(columns, value, mode):
            
            if isinstance(val, tuple) and len(val) == 2:
                mod == "range"
                value_removed_df = value_removed_df[(value_removed_df[col] >= val[0]) & (value_removed_df[col] <= val[1])]
            elif isinstance(val, int):
                if mod == "below":
                    value_removed_df = value_removed_df[value_removed_df[col] <= val]
                elif mod == "above":
                    value_removed_df = value_removed_df[value_removed_df[col] >= val]
                else:
                    raise ValueError("""Please type "above", "below" or "range" for the mode to start removing""")

            else:
                raise ValueError(f"Invalid value type for column '{col}'. Must be an int or tuple.")
            
        return value_removed_df
    
    def dup_row_remover(self, df: pd.DataFrame, columns:Union[str,list] = None, keep: str = "first", error_skip: bool = False) -> pd.DataFrame:
        
        """    
        subset (list or str, optional): Column label(s) to consider for identifying duplicates.
                            If None, considers all columns.
        
        keep (str, optional): Determines which duplicates to keep.
                          'first' (default) - keeps first occurrence
                          'last' - keeps last occurrence
                          'False' - drops all duplicates
        """        
        if isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col not in df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")
        
        if keep not in ["first", "last", False]:
            raise ValueError("'keep' should be either 'first', 'last', or False. Try again.")

        dup_row_df = df.copy()
        
        dup_row_df = dup_row_df.drop_duplicates(subset=columns, keep= keep).reset_index(drop = True)
        return dup_row_df

    def missing_rows(self, df: pd.DataFrame, drop_threshold: float = None, axis: str = None, inplace: bool = False) -> pd.DataFrame:
        """
        drop_threshold : float, optional
            The maximum allowed percentage of missing values before dropping.
            Must be between 0 and 1. For example, 0.3 means drop rows/columns with more than 30% missing values.
        axis : {'rows', 'columns'}, optional
            Whether to drop rows or columns exceeding the threshold. Must be 'rows' or 'columns'.
        inplace : bool, optional
            Whether to apply the changes to the original DataFrame or return a new one.

        Returns:
        -------
        pd.DataFrame
            The DataFrame after applying the optional drop operation.
        """

        print("="*40)
        print("📊 Missing Data Insight")
        print("="*40)

        total_missing = df.isnull().sum()
        percent_missing = (total_missing / len(df)) * 100
        missing_report = pd.DataFrame({
            'Total Missing': total_missing,
            'Percent Missing': percent_missing.round(2)
        })
        
        print(missing_report[missing_report['Total Missing'] > 0])

        if drop_threshold is not None and axis is not None:
            
            if not 0 <= drop_threshold <= 1:
                raise ValueError("drop_threshold must be between 0 and 1.")
            
            if axis not in ['rows', 'columns']:
                raise ValueError("axis must be either 'rows' or 'columns'.")

            print(f"\n⚠️ Applying drop operation for {axis} with more than {drop_threshold*100:.2f}% missing values...")

            if axis == 'rows':
                condition = df.isnull().mean(axis=1) > drop_threshold
                num_to_drop = condition.sum()
                print(f"Dropping {num_to_drop} rows.")
                
                if inplace:
                    df.drop(df[condition].index, inplace=True)
                else:
                    df = df.loc[~condition]

            elif axis == 'columns':
                condition = df.isnull().mean(axis=0) > drop_threshold
                num_to_drop = condition.sum()
                print(f"Dropping {num_to_drop} columns.")
                
                if inplace:
                    df.drop(columns=df.columns[condition], inplace=True)
                else:
                    df = df.loc[:, ~condition]

            return df
        return df
    
    def convert_case(self, df: pd.DataFrame, columns: Union[list, str], mode: str, error_skip: bool = False) -> pd.DataFrame:
        """
        Convert the case of text in specified columns.

        Parameters:
            df (pd.DataFrame): The DataFrame to process.
            columns (list): List of column names to apply the case conversion.
            mode (str): One of 'lower', 'upper', 'title', or 'capitalize'.

        Returns:
            pd.DataFrame: DataFrame with updated case in specified columns.
        """
        mode = mode.lower()
        valid_modes = {'lower', 'upper', 'title', 'capitalize'}
        
        if mode not in valid_modes:
            raise ValueError(f"Unsupported case conversion mode: {mode}")

        if isinstance(columns, str):
            columns = [columns]
        
        case_convert_df = df.copy()
        
        for col in columns:
            if col not in case_convert_df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")
            
            if not pd.api.types.is_string_dtype(case_convert_df[col]):
                case_convert_df[col] = case_convert_df[col].astype(str)

            if mode == 'lower':
                case_convert_df[col] = case_convert_df[col].str.lower()
            elif mode == 'upper':
                case_convert_df[col] = case_convert_df[col].str.upper()
            elif mode == 'title':
                case_convert_df[col] = case_convert_df[col].str.title()
            elif mode == 'capitalize':
                case_convert_df[col] = case_convert_df[col].apply(lambda x: ' '.join([word.capitalize() for word in x.split()]))

        return case_convert_df

    def remove_whitespace(self, df: pd.DataFrame, columns: Union[list, str], error_skip: bool = False) -> pd.DataFrame:
        """
        Remove unnecessary whitespace from string columns:
        - Leading/trailing spaces
        - Extra spaces within the text

        Parameters:
            df (pd.DataFrame): The DataFrame to process.
            columns (list): List of column names to clean.

        Returns:
            pd.DataFrame: DataFrame with cleaned whitespace in specified columns.
        """
        if isinstance (columns, str):
            columns = [columns]
        
        whitespace_df = df.copy()
        
        for col in columns:
            if col not in whitespace_df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")
            
            if not pd.api.types.is_string_dtype(whitespace_df[col]):
                whitespace_df[col] = whitespace_df[col].astype(str)
            
            whitespace_df[col] = whitespace_df[col].apply(
                lambda x: ' '.join(x.strip().split()) if isinstance(x, str) else x
            )
        
        return whitespace_df

    def standardize_dates(self, df: pd.DataFrame, columns: Union[list, str], output_format: str = "%Y-%m-%d", error_skip: bool = False) -> pd.DataFrame:
        """
        Standardize date formats in specified columns to a uniform format.

        Parameters:
            df (pd.DataFrame): The DataFrame to process.
            columns (list): List of column names to convert.
            output_format (str): The desired date format (default is 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: DataFrame with standardized date formats.
        """
        if isinstance (columns, str):
            columns = [columns]
        
        standardize_df = df.copy()
        
        for col in columns:
            if col not in standardize_df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")
            
            def parse_date(val):
                if pd.isna(val):
                    return pd.NaT
                try:
                    return parse(str(val), dayfirst=False).strftime(output_format)
                except (ValueError, TypeError):
                    return val  # return original if not parseable
            
            standardize_df[col] = standardize_df[col].apply(parse_date)

        return standardize_df
    
    def format_numbers(self, df: pd.DataFrame, columns: Union[list, str], decimal_places: int = 2, drop_invalid: bool = False, error_skip : bool = False ) -> pd.DataFrame:
        """
        Format numeric values in specified columns:
        - Remove currency symbols and commas
        - Convert to float with specific decimal places
        - Optionally drop rows with unconvertible values

        Args:
            df (pd.DataFrame): The input DataFrame
            columns (list[str]): List of column names to process
            decimal_places (int): Number of decimal places to round to
            drop_invalid (bool): Whether to drop rows with invalid values (default: False)

        Returns:
            pd.DataFrame: The updated DataFrame with formatted numbers
        """
        if isinstance (columns, str):
            columns = [columns]
        
        number_format_df = df.copy()
        for col in columns:
            if col not in number_format_df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")
            
            # Remove currency symbols, commas, and whitespace
            number_format_df[col] = (
                number_format_df[col]
                .astype(str)
                .str.replace(r'[^0-9\.-]', '', regex=True)
            )

            # Convert to numeric, forcing errors to NaN
            number_format_df[col] = pd.to_numeric(number_format_df[col], errors='coerce')

            # Round to desired decimal places
            number_format_df[col] = number_format_df[col].round(decimal_places)

            # Drop or warn about invalid rows
            if drop_invalid:
                number_format_df = number_format_df[number_format_df[col].notna()]

        return number_format_df

    def normalize_categories(self, df:pd.DataFrame, columns: Union[list, str], mapping, threshold=80, error_skip=False):
        """
        Normalize categorical values in a DataFrame column to consistent names using fuzzy matching.

        Parameters:
            df (pd.DataFrame): The DataFrame.
            column (str): The column name to normalize.
            mapping (dict): Canonical name -> list of known variations.
            threshold (int): Minimum match score (0-100) for replacement.
            error_skip (bool): If True, skip unmatched values; else raise KeyError.

        Returns:
            pd.DataFrame: DataFrame with normalized categories.
        """
        
        if isinstance (columns, str):
            columns = [columns]
            
        # Build a list of all possible values and their canonical forms
        reference_dict = {}
        for canonical, variants in mapping.items():
            for variant in variants + [canonical]:
                reference_dict[variant.lower()] = canonical

        all_variants = list(reference_dict.keys())

        def normalize_value(val):
            if pd.isna(val):
                return val
            val_str = str(val).lower()
            if val_str in reference_dict:
                return reference_dict[val_str]
            match, score, _ = process.extractOne(val_str, all_variants, scorer=fuzz.token_sort_ratio)
            if score >= threshold:
                return reference_dict[match]
            elif error_skip:
                return val
            else:
                raise KeyError(f"No match found for '{val}' (best score: {score})")
        
        category_normalize_df = df.copy()
        
        for col in columns:
            category_normalize_df[col] = category_normalize_df[col].apply(normalize_value)

        return category_normalize_df

    def remove_irrelevant_characters(self, df: pd.DataFrame, columns: Union[list, str], error_skip: bool = True):
        """
        Remove irrelevant characters such as HTML tags and special symbols 
        from the specified DataFrame columns.

        Args:
            df (pd.DataFrame): The DataFrame to modify.
            columns (str | list[str]): Column name(s) to clean.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        # Ensure columns is a list
        if isinstance(columns, str):
            columns = [columns]

        irrelevant_df = df.copy()
        
        for col in columns:
            if col not in irrelevant_df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")

            irrelevant_df[col] = irrelevant_df[col].apply(
                lambda x: (
                    re.sub(r'\s+', ' ', 
                        re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', 
                            re.sub(r'<.*?>', '', x)
                        )
                    ).strip()
                ) if isinstance(x, str) else x
            )

        return irrelevant_df
    
    def normalize(self, df: pd.DataFrame,
                  columns: Union[list, str],
                  mapping_return: bool = False,
                  mapping_format: str = "joblib",
                  save_scaler: bool = True,
                  error_skip : bool = False) -> pd.DataFrame:
        
        # Convert a single column name to a list for consistent processing
        if isinstance(columns, str):
            columns = [columns]

        normalize_df = df.copy()
        
        # Check if all specified columns exist in the DataFrame
        for col in columns:
            if col not in normalize_df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")

        # Initialize the scaler and mapping manager
        mapping_manager = MappingManager() if mapping_return else None
        
        # Loop through each column to apply normalization
        for col in columns:
            
            scaler = MinMaxScaler()
            # Perform Min-Max Scaling and reshape to fit the scaler's expected input
            normalized = scaler.fit_transform(normalize_df[[col]])
            
            if save_scaler:
                save_path_scaler = "scalers"
                os.makedirs(save_path_scaler, exist_ok=True)
                file_path = os.path.join(save_path_scaler, f"{col}_scaler.pkl")
                joblib.dump(scaler, file_path)
                print(f"✅ Scaler saved: {file_path}")

            # Save the scaler object as mapping if requested
            if mapping_return:
                if mapping_format.lower().strip() == "json":
                    mapping_manager.save_mapping(col, scaler, format="json")
                
                else:
                    mapping_manager.save_mapping(col, scaler, format="joblib")

            # Update the DataFrame column with the normalized values
            normalize_df[col] = normalized

        return normalize_df
    
    def unique_items_list(self, df: pd.DataFrame, columns: Union[list, str], count: bool = False, error_skip: bool = False):
        
        result = {}  # Dictionary to store unique items for each column

        if isinstance(columns, str):
            columns = [columns]
        
        # Check if columns exist in the DataFrame
        for col in columns:
            if col not in df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")
        
        for col in columns:
            if col in df.columns:  # Ensure the column exists in the DataFrame
                unique_items = df[col].dropna().astype(str).str.strip().unique()
                result[col] = sorted(unique_items)  # Sort the unique items alphabetically
        
        if count == True:
            for col in result:
                print(f"{col}: {len(result[col])}")
        else:
            pass
        
        print(result)
        return result

    def min_max_finder(self, df: pd.DataFrame, columns: Union[list, str], error_skip: bool = False) -> list:
          
        if isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col not in df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")
        
        for col in columns:
            print(f"{col}:")
            print(df[col].agg(['min', 'max']))
            print()

    def OneHotEncoder(self, df: pd.DataFrame, columns: Union[list, str], mapping_return: bool = False, error_skip: bool = False) -> pd.DataFrame:
        
        # Convert a single column name to a list for consistent processing
        if isinstance(columns, str):
            columns = [columns]

        OneHot_df = df.copy()
        
        # Check if all specified columns exist in the DataFrame
        for col in columns:
            if col not in OneHot_df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")

        # Initialize the OneHotEncoder with settings to handle unknown and missing values
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform="pandas")
        
        # Initialize the mapping manager
        mapping_manager = MappingManager() if mapping_return else None
        
        # Loop through each column to apply one-hot encoding
        for col in columns:
            # Transform the data using the encoder and store as a DataFrame
            ohe_transformed = ohe.fit_transform(OneHot_df[[col]])

            # Save the fitted encoder object as mapping if requested
            if mapping_return:
                mapping_manager.save_mapping(col, ohe)

            # Concatenate the one-hot encoded columns with the original DataFrame
            OneHot_df = pd.concat([OneHot_df.drop(col, axis=1), ohe_transformed], axis=1)

        return OneHot_df


    def TargetEncoder(self, df: pd.DataFrame, columns: Union[list, str], target: str, mapping_return: bool = False, error_skip : bool = False) -> pd.DataFrame:
        # Convert a single column name to a list for consistent processing
        if isinstance(columns, str):
            columns = [columns]

        Target_df = df.copy()
        # Check if all specified columns exist in the DataFrame
        for col in columns:
            if col not in Target_df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")

        Target_df[target] = pd.to_numeric(Target_df[target], errors="coerce")
        Target_df = Target_df.dropna(subset=[target])
        
        # Initialize the TargetEncoder with configurations
        te = TargetEncoder(smoothing=4.0, handle_unknown="value", min_samples_leaf=10.0, handle_missing="value")

        # Initialize the mapping manager
        mapping_manager = MappingManager() if mapping_return else None
        
        # Apply target encoding using the specified columns and target variable
        te_transformed = te.fit_transform(Target_df[columns], pd.Series(Target_df[target], name=target))

        # Save the fitted encoder object as mapping if requested
        if mapping_return:
            mapping_manager.save_mapping(",".join(columns), te)

        # Merge the target encoded columns back into the original DataFrame
        Target_df = pd.concat([Target_df.drop(columns, axis=1), te_transformed], axis=1)

        return Target_df

    
    def FrequencyEncoder(self, df: pd.DataFrame, columns: Union[list, str], mapping_return: bool = False, error_skip: bool = False) -> pd.DataFrame:
        
        # Convert a single column name to a list for consistent processing
        if isinstance(columns, str):
            columns = [columns]

        freq_df = df.copy()
        
        # Check if all specified columns exist in the DataFrame
        for col in columns:
            if col not in freq_df.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' does not exist in the DataFrame. Continuing Operation...")
                    continue
                else:
                    raise KeyError(f"⚠️ Column '{col}' does not exist in the DataFrame.")

        # Initialize the mapping manager
        mapping_manager = MappingManager() if mapping_return else None
        
        for col in columns:
            # Calculate frequency of each unique value
            freq_df[f"{col}_freq"] = freq_df[col].map(freq_df[col].value_counts(normalize=True))
            freq = freq_df[col].value_counts(normalize=True).to_dict()

            # Save the frequency mapping if requested
            if mapping_return:
                mapping_manager.save_mapping(col, freq)

            # Map each value to its calculated frequency
            freq_df[col] = freq_df[col].map(freq)

        return freq_df


    def save_dataframe(self, df, output_path: str, file_format: str):
        """Write the cleaned dataset to a new file for future use."""
        
        if file_format.lower().strip() == "csv":
            df.to_csv(output_path, index = False)
            print(f"Data saved to {output_path}")
            
        elif file_format.lower().strip() == "json":
            df.to_json(output_path, index = False)
            print(f"Data saved to {output_path}")
            
        elif file_format.lower().strip() == "jsonl":
            df.to_json(output_path, index = False, lines = True, orient = "records")
            print(f"Data saved to {output_path}")
            
        elif file_format.lower().strip() == "excel":
            df.to_excel(output_path, index = False)
            print(f"Data saved to {output_path}")
        
        else:
            print("please enter a viable file format (CSV, JSON, JSONL, Excel)")
            
        return df