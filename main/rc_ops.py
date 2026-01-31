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
    
    def drop_columns(
        self,
        engine,
        table: str,
        columns: Union[str, List[str]],
        error_skip: bool = False,
        schema: str = "public"
    ) -> Dict:

        # Normalize input
        if isinstance(columns, str):
            columns = [columns]

        with engine.begin() as conn:
            # Inspect existing columns
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

            dropped = []

            for col in columns:
                if col not in existing_cols:
                    print(f"Column(s) '{col}' Not Found. Continuing Operation...")
                    continue

                conn.execute(text(f"""
                    ALTER TABLE {schema}.{table}
                    DROP COLUMN {col};
                """))

                dropped.append(col)
        
        print("✅ Columns Dropped")

        return {
            "operation": "drop_columns",
            "table": f"{schema}.{table}",
            "dropped_columns": dropped,
            "skipped_columns": missing,
            "status": "success"
        }
    
    def value_remover(
        self, 
        engine,
        table:str,
        value: Union[int, tuple],
        columns: Union[str, list], 
        mode: Union[str.lower, list],
        schema: str = "public", 
        error_skip: bool = False
    )-> Dict:
        
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(value, int) or isinstance(value, tuple):
            value = [value]
        if isinstance(mode, str):
            mode = [mode]
        
        if not (len(columns) == len(value) == len(mode)):
            raise ValueError("Columns, values, and modes must have the same length.")
        
        mode = [m.lower() for m in mode]
        
        with engine.begin() as conn:
            # Inspect existing columns
            result = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND table_name = :table;
            """), {"schema": schema, "table": table})
            
            existing_cols = {row[0] for row in result}

            missing = [c for c in columns if c not in existing_cols]
            if missing and not error_skip:
                raise KeyError(f"Missing columns: {missing}")
            
            operations = []
            total_rows_removed = 0
            
            for col, val, mod in zip(columns, value, mode):
                
                if col not in existing_cols:
                    continue
                
                if isinstance(val, tuple):
                    
                    if len(val) != 2 or mod not in ("outside", "between"):
                        raise ValueError(
                            f"Column'{col}': tuple values require mode ='outside' or 'between'"
                        )
                    
                    if mod == "outside":
                        results = conn.execute(text(f"""
                            DELETE FROM {schema}.{table}
                            WHERE {col} NOT BETWEEN :low AND :high;
                            """),
                            {"low": val[0], "high": val[1]}
                        )
                        
                        print(f"✅ Values Outside {val[0]} And {val[1]} Removed From {col}")
                    
                    elif mod== "between":
                        results = conn.execute(text(f"""
                            DELETE FROM {schema}.{table}
                            WHERE {col} BETWEEN :low AND :high;
                            """),
                            {"low": val[0], "high": val[1]}
                        )
                        
                        print(f"✅ Values Between {val[0]} And {val[1]} Removed From {col}")

                elif isinstance(val, int):
                    
                    if mod == "below":
                        results = conn.execute(text(f"""
                            DELETE FROM {schema}.{table}
                            WHERE {col} < :val;
                        """),
                        {"val": val})
                        
                        print(f"✅ Values Under {val} Removed")

                    elif mod == "above":
                        results = conn.execute(text(f"""
                            DELETE FROM {schema}.{table}
                            WHERE {col} > :val;
                        """),
                        {"val": val})
                        
                        print(f"✅ Values Above {val} Removed")
                        
                    elif mod == "exact":
                        results = conn.execute(text(f"""
                            DELETE FROM {schema}.{table}
                            WHERE {col} = :val;
                        """),
                        {"val": val})
                        
                        print("✅ Removed The Requested Values")
                        
                    else:
                        raise ValueError(
                            f"invalid mode ('{mod}') for column '{col}'"
                            )

                else:
                    raise ValueError(
                        f"Invalid value type for column '{col}'. Must be an int or tuple."
                        )
            
            rows_removed = results.rowcount
            total_rows_removed += rows_removed
            operations.append({
                "column": col,
                "mode": mod,
                "value": val,
                "rows_removed": rows_removed
            })
            
        return {
        "operation": "value_remover",
        "table": f"{schema}.{table}",
        "operations": operations,
        "total_rows_removed": total_rows_removed,
        "skipped_columns": missing,
        "status": "success"
    }
    
    def dup_row_remover(
        self, 
        engine,
        table:str,
        columns: Union[str, list], 
        keep: Union[str, bool],
        schema: str = "public", 
        error_skip: bool = False
    )-> Dict:
        
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
        
        if keep not in ["first", "last", False]:
            raise ValueError("'keep' should be either 'first', 'last', or False. Try again.")

        with engine.begin() as conn:
            # Inspect existing columns
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

            operations = []
            total_rows_removed = 0

            valid_columns = [c for c in columns if c in existing_cols]

            if not valid_columns:
                raise ValueError("No valid columns left to check for duplicates.")
            
            partition_cols = ", ".join(valid_columns)

            if keep == "first":
                results = conn.execute(text(f"""
                    DELETE FROM {schema}.{table}
                    WHERE id IN (
                        SELECT id
                        FROM (
                            SELECT id,
                                ROW_NUMBER() OVER (
                                    PARTITION BY {partition_cols}
                                    ORDER BY id ASC
                                ) AS rn
                            FROM {schema}.{table}
                        ) sub
                        WHERE rn > 1
                    );"""))

                print("✅ Duplicate rows removed. First one kept.")
                    
            elif keep == "last":
                results = conn.execute(text(f"""
                    DELETE FROM {schema}.{table}
                    WHERE id IN (
                        SELECT id
                        FROM (
                            SELECT id,
                                ROW_NUMBER() OVER (
                                    PARTITION BY {partition_cols}
                                    ORDER BY id DESC
                                ) AS rn
                            FROM {schema}.{table}
                        ) sub
                        WHERE rn > 1
                    );"""))
                    
                print("✅ Duplicate rows removed. Last one kept.")
                    
            elif keep == False:
                results = conn.execute(text(f"""
                    DELETE FROM {schema}.{table}
                    WHERE ({partition_cols}) IN (
                        SELECT {partition_cols}
                        FROM {schema}.{table}
                        GROUP BY {partition_cols}
                        HAVING COUNT(*) > 1
                    );"""))
                    
                print("✅ All duplicate rows removed.")
                    
        
            rows_removed = results.rowcount
            total_rows_removed += rows_removed
            operations.append({
                "columns": valid_columns,
                "keep": keep,
                "rows_removed": rows_removed
            })

            
        return {
        "operation": "dup_row_remover",
        "table": f"{schema}.{table}",
        "operations": operations,
        "total_rows_removed": total_rows_removed,
        "skipped_columns": missing,
        "status": "success"
        }