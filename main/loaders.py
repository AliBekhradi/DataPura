import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
import psycopg2
import tempfile

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

    def refresh_table(self, 
            file_path: str,
            table_name: str,
            db_config: dict):
            
            """
            Format-agnostic loader.
            Accepts CSV, JSON, JSONL, Excel.
            Converts to temporary CSV, sends to PostgreSQL.
            """

            ext = os.path.splitext(file_path)[1].lower()

            try:
                # 2. Read file into a DataFrame
                if ext == ".csv":
                    df = pd.read_csv(file_path)

                elif ext == ".json":
                    df = pd.read_json(file_path)

                elif ext == ".jsonl":
                    df = pd.read_json(file_path, lines=True)

                elif ext in (".xlsx", ".xls"):
                    df = pd.read_excel(file_path)

                else:
                    raise ValueError(f"Unsupported file format: {ext}")

                if df.empty:
                    raise ValueError("Input dataset is empty.")

                # 3. Convert DataFrame to temporary CSV
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    temp_csv_path = tmp.name
                df.to_csv(temp_csv_path, index=False)

                # 4. Load CSV into PostgreSQL
                conn = psycopg2.connect(**db_config)
                cur = conn.cursor()

                cur.execute(f"TRUNCATE TABLE {table_name};")

                with open(temp_csv_path, "r", encoding="utf-8") as f:
                    cur.copy_expert(
                        f"COPY {table_name} FROM STDIN WITH CSV HEADER",
                        f
                    )

                conn.commit()
                print(f"✅ Loaded '{file_path}' into '{table_name}'")

            except FileNotFoundError:
                print(f"❌ File not found: {file_path}")

            except Exception as e:
                print(f"❌ Error during load: {e}")

            finally:
                # 5. Cleanup temp CSV and close DB
                if "conn" in locals():
                    conn.close()
                if "temp_csv_path" in locals() and os.path.exists(temp_csv_path):
                    os.remove(temp_csv_path)    
        
    def save_dataframe(df,
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