import pandas as pd
from typing import Union, List, Dict
import numpy as np
from sqlalchemy import text

class RowsAndColumns:
    
    def __init__(self):
        pass
    
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
        
        print("✅ Imputation Complete")
        return imputed_df
    
    def rename_columns(
        self,
        engine,
        table: str,
        columns: Union[str, List[str]],
        rename_to: Union[str, List[str]],
        error_skip: bool = False,
        schema: str = "public"
    ) -> Dict:

        # Normalize inputs
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(rename_to, str):
            rename_to = [rename_to]

        if len(columns) != len(rename_to):
            raise ValueError("Length of columns and rename_to must match.")

        with engine.begin() as conn:
            # Fetch existing columns
            result = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND table_name = :table
            """), {"schema": schema, "table": table})

            existing_cols = {row[0] for row in result}

            missing = [c for c in columns if c not in existing_cols]
            if missing and not error_skip:
                raise KeyError(f"Missing columns: {missing}")

            applied = {}

            for old, new in zip(columns, rename_to):
                if old not in existing_cols:
                    continue

                conn.execute(text(f"""
                    ALTER TABLE {schema}.{table}
                    RENAME COLUMN {old} TO {new};
                """))

                applied[old] = new
        
        print("✅ Columns Renamed")
        
        return {
            "operation": "rename_columns",
            "table": f"{schema}.{table}",
            "applied_renames": applied,
            "skipped_columns": missing,
            "status": "success"
        }
    
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
        
        print("✅ Columns Dropped")
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
                
                if mod == "range":
                    value_removed_df = value_removed_df[(value_removed_df[col] >= val[0]) & (value_removed_df[col] <= val[1])]
                    print(f"✅ Values Outside {val[0]} And {val[1]} Removed")
                else:
                    raise ValueError("Tuple values require mode='range'")
            
            elif isinstance(val, int):
                
                if mod == "below":
                    value_removed_df = value_removed_df[value_removed_df[col] <= val]
                    print(f"✅ Values Under {val} Removed")
                elif mod == "above":
                    value_removed_df = value_removed_df[value_removed_df[col] >= val]
                    print(f"✅ Values Above {val} Removed")
                elif mod == "exact":
                    value_removed_df[col] = value_removed_df[col].progress_apply(lambda x: x.replace(val, pd.NA))
                    value_removed_df = value_removed_df.dropna(subset=col)
                    print("✅ Removed The Requested Values")
                else:
                    raise ValueError("""Please type "above", "below", "exact" or "range" for the mode to start removing""")

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
        
        print("✅ Duplicate Rows Removed")
        return dup_row_df