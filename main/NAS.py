import pandas as pd
from typing import Union, Dict
import os
import joblib
from sqlalchemy import text
from sklearn.preprocessing import MinMaxScaler
from dateutil.parser import parse
from rapidfuzz import process, fuzz
from mapping import MappingManager


class NAS:
    
    def __init__(self):
        pass
    
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
        
        print("✅ Normalization Complete")
        return normalize_df
    
    def standardize_dates(
        self,
        engine,
        table: str,
        df: pd.DataFrame,
        columns: Union[list, str],
        schema: str = "public",
        error_skip: bool = False
    ) -> Dict:
        """
        Standardize date columns using Python parsing, then enforce
        correctness and performance guarantees in SQL.
        """

        if isinstance(columns, str):
            columns = [columns]

        df_out = df.copy()
        operations = []
        
        for col in columns:
            if col not in df_out.columns:
                if error_skip:
                    print(f"⚠️ Column '{col}' not found. Skipping.")
                    continue
                else:
                    raise KeyError(f"Column '{col}' does not exist.")

            def parse_date(val):
                if pd.isna(val):
                    return None
                try:
                    return parse(str(val), dayfirst=False).date()
                except (ValueError, TypeError):
                    return val  # preserve original if unparseable

            before_nulls = df_out[col].isna().sum()
            df_out[col] = df_out[col].apply(parse_date)
            after_nulls = df_out[col].isna().sum()

            operations.append({
                "column": col,
                "parsed_nulls_added": after_nulls - before_nulls
            })

        df_out.to_sql(
            table,
            engine,
            schema=schema,
            if_exists="replace",
            index=False
        )

        with engine.begin() as conn:
            for col in columns:
                # Enforce DATE type
                conn.execute(text(f"""
                    ALTER TABLE {schema}.{table}
                    ALTER COLUMN {col} TYPE DATE
                    USING {col}::DATE;
                """))

                # Enforce sane bounds
                conn.execute(text(f"""
                    ALTER TABLE {schema}.{table}
                    ADD CONSTRAINT {table}_{col}_valid_date
                    CHECK (
                        {col} BETWEEN DATE '1900-01-01' AND CURRENT_DATE
                    );
                """))

                # Index for performance
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS
                    idx_{table}_{col}
                    ON {schema}.{table} ({col});
                """))
        
        print("✅ Dates Standardized Succesfully")
        
        return {
            "operation": "standardize_dates",
            "table": f"{schema}.{table}",
            "columns": columns,
            "python_parsing": True,
            "sql_constraints_applied": True,
            "indexes_created": True,
            "operations": operations,
            "status": "success"
        }
    
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
        
        print("✅ Numbers Formatted Succesfully")
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

        print("✅ Categories Normalized Succesfully")
        return category_normalize_df