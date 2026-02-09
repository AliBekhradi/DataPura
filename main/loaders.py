import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
from pathlib import Path
import re
from datetime import datetime

class DataLoader:
    engine = create_engine("postgresql://admin:10293847@localhost:5432/datapura")
    
    def __init__(self):
        pass
    
    def ingest(self, file_path: str, return_df: bool = False, preview: bool = False):
        table_name = os.path.splitext(os.path.basename(file_path))[0].lower()
        ext = os.path.splitext(file_path)[1].lower()

        result_df = None

        if ext in [".csv", ".json", ".jsonl", ".xlsx", ".xls"]:
            if ext == ".csv":
                df = pd.read_csv(file_path, header=0)
            elif ext == ".json":
                df = pd.read_json(file_path)
            elif ext == ".jsonl":
                df = pd.read_json(file_path, lines=True)
            else:
                df = pd.read_excel(file_path)

            if df.empty:
                raise ValueError("Loaded DataFrame is empty.")

            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(r"[^a-z0-9_]+", "_", regex=True)
            )
            
            has_id = "id" in df.columns

            if has_id:
                print(f"ID column found. Proceeding with table creation...")
                df.to_sql(table_name, DataLoader.engine, if_exists="replace", index=False)
            
            if return_df:
                result_df = pd.read_sql(
                    f"SELECT * FROM {table_name}", DataLoader.engine
                )
            
            else:
                DTYPE_MAP = {
                    "int": "BIGINT",
                    "float": "DOUBLE PRECISION",
                    "datetime": "TIMESTAMP",
                    "bool": "BOOLEAN",
                }

                column_defs = []

                for col, dtype in zip(df.columns, df.dtypes):
                    sql_type = "TEXT"
                    dtype_str = str(dtype)

                    for key, mapped_type in DTYPE_MAP.items():
                        if key in dtype_str:
                            sql_type = mapped_type
                            break

                    column_defs.append(f"{col} {sql_type}")

                columns_sql = ",\n    ".join(column_defs)

                create_table_sql = f"""
                DROP TABLE IF EXISTS {table_name};

                CREATE TABLE {table_name} (
                    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    {columns_sql}
                );
                """
                
                with DataLoader.engine.begin() as conn:
                    conn.execute(text(create_table_sql))
                
                df.to_sql(table_name, DataLoader.engine, if_exists="append", index= False)
    
        elif ext == ".sql":
            with open(file_path, "r", encoding="utf-8") as f:
                sql_text = f.read()

            statements = [
                s.strip()
                for s in sql_text.split(";")
                if s.strip() and not s.strip().startswith(("--", "/*"))
            ]

            if not statements:
                return pd.DataFrame() if return_df else None

            try:
                with DataLoader.engine.begin() as conn:
                    for stmt in statements:
                        conn.execute(text(stmt))
            except Exception as e:
                raise RuntimeError(
                    f"Failed executing SQL file {file_path}. Details: {e}"
                ) from e

            if return_df:
                try:
                    result_df = pd.read_sql(
                        f"SELECT * FROM {table_name}", DataLoader.engine
                    )
                except Exception:
                    print(
                        f"Warning: SQL script executed, but could not read table '{table_name}'."
                    )
                    result_df = pd.DataFrame()
            
        else:
            raise ValueError(f"Unsupported format: {ext}")

        print("✅ File ingested and available in the main database")
        
        if preview:
            preview_df = (
                result_df
                if result_df is not None
                else pd.read_sql(f"SELECT * FROM {table_name}", DataLoader.engine)
            )
            self._print_df_info(preview_df)
        
        return result_df
    
    def read_from_sql(self, table_name: str) -> list[str]:
        """
        Reads the current schema of a table from the database.
        Used for validating SQL operations (drop/rename/etc).
        """

        query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = :table_name
            ORDER BY ordinal_position
        """

        try:
            with DataLoader.engine.begin() as conn:
                result = conn.execute(
                    text(query),
                    {"table_name": table_name}
                )
                columns = [row[0] for row in result]
        except Exception as e:
            raise RuntimeError(
                f"Failed reading schema for table '{table_name}'."
            ) from e

        if not columns:
            raise RuntimeError(
                f"Table '{table_name}' exists but has no columns."
            )

        return columns

    def _print_df_info(self, df: pd.DataFrame):
        print("=" * 50)
        print("📊 Dataset Overview")
        print("=" * 50)

        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(list(df.columns))

        print("\nDtypes:")
        print(df.dtypes)

        print("\nMissing Values (%):")
        print((df.isna().mean() * 100).round(2))

        print("\nPreview:")
        print(df.head(3))

        print("=" * 50)

    def export_to_csv(
        self,
        table_name: str,
        output_path: str | None = None,
    ):
        if output_path is None:
            output_path = f"{table_name}.csv"

        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", DataLoader.engine)
        except Exception as e:
            raise RuntimeError(
                f"Failed to export table '{table_name}'. Table may not exist."
            ) from e

        df.to_csv(output_path, index=False)

        print(f"✅ Table '{table_name}' exported to {output_path}")

        return output_path
    
    def strip_run_id(self, name: str) -> str:
        return re.sub(r"_\d{8}_\d{6}$", "", name)

    def apply_table_update(self,
            table_name: str,
            df_name: str = None,
            file_path: str = None,
            schema: str = "public",
            replace: bool = False):
            
            BASE_DIR = Path(__file__).resolve().parent
            source_file = Path(file_path) if file_path else (BASE_DIR / df_name if df_name else None)
            
            df = pd.read_csv(source_file, header=0)
            df.columns = [col.lower() for col in df.columns]
            df_cols = df.columns
            columns_sql = ", ".join(df_cols)

            if not source_file or not source_file.exists():
                raise FileNotFoundError(f"Missing or invalid file: {source_file}")
            
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name) or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", schema):
                raise ValueError(f"Unsafe SQL identifier: Check the table name or schema name")
            
            # 2. DATE ID GENERATION Using Year-Month-Day-Hour-Minute-Second for safety
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            with DataLoader.engine.begin() as conn:
                cur = conn.connection.cursor()

                if replace:
                    # Construct paths once to avoid typos
                    live_path = f"{schema}.{table_name}"
                    tmp_name = f"_tmp_{table_name}"
                    tmp_path = f"{schema}.{tmp_name}"
                    old_name = f"_old_{table_name}"
                    old_path = f"{schema}.{old_name}"

                    conn.execute(text(f"DROP TABLE IF EXISTS {tmp_path};"))
                    conn.execute(text(f"CREATE TABLE {tmp_path} (LIKE {live_path} INCLUDING ALL);"))
                    missing = set(df_cols) - set(pd.read_sql(f"SELECT * FROM {schema}.{table_name} LIMIT 0", conn).columns)
                    if missing:
                        raise ValueError(f"CSV has columns not in table: {missing}")

                    with open(source_file, "r", encoding="utf-8") as f:
                        cur.copy_expert(
                        f"COPY {tmp_path} ({columns_sql}) FROM STDIN WITH CSV HEADER;",f)

                    conn.execute(text(f"DROP TABLE IF EXISTS {old_path};"))
                    # RENAME TO takes a NAME, not a PATH (No schema allowed after 'TO')
                    conn.execute(text(f"ALTER TABLE {live_path} RENAME TO {old_name};"))
                    conn.execute(text(f"ALTER TABLE {tmp_path} RENAME TO {table_name};"))
                    conn.execute(text(f"DROP TABLE {old_path};"))
                    
                    print(f"Table '{table_name}' successfully updated in schema '{schema}'.")

                else:
                    # --- UNIQUE DATE LOGIC ---
                    base_table = self.strip_run_id(table_name)
                    target_table_name = f"{base_table}_{run_id}"
                    # FIXED: Ensure the new dated table is created in the same schema
                    target_table_path = f"{schema}.{target_table_name}"
                    
                    conn.execute(text(f"CREATE TABLE {target_table_path} (LIKE {schema}.{table_name} INCLUDING ALL);"))
                    
                    missing = set(df_cols) - set(pd.read_sql(f"SELECT * FROM {schema}.{table_name} LIMIT 0", conn).columns)
                    if missing:
                        raise ValueError(f"CSV has columns not in table: {missing}")

                    with open(source_file, "r", encoding="utf-8") as f:
                        cur.copy_expert(
                        f"COPY {target_table_path} ({columns_sql}) FROM STDIN WITH CSV HEADER;",f)
                        
                    print(f"New snapshot created: {target_table_path}")
        
    def save_dataframe(self,
                    df,
                    output_path: str,
                    file_format: str):
        """Write the cleaned dataset to a new file for future use."""
            
        if file_format.lower().strip() == "csv":
            df.to_csv(output_path, index = False)
            print(f"✅ Data Succesfully Saved To: {output_path}")
                
        elif file_format.lower().strip() == "json":
            df.to_json(output_path, index = False)
            print(f"✅ Data Succesfully Saved To: {output_path}")
                
        elif file_format.lower().strip() == "jsonl":
            df.to_json(output_path, index = False, lines = True, orient = "records")
            print(f"✅ Data Succesfully Saved To: {output_path}")
                
        elif file_format.lower().strip() == "excel":
            df.to_excel(output_path, index = False)
            print(f"✅ Data Succesfully Saved To: {output_path}")
            
        else:
            print("please enter a viable file format (CSV, JSON, JSONL, Excel)")
                
        return df