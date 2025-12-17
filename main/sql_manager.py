import os
import pandas as pd
import json
from sqlalchemy import create_engine
from sqlalchemy import text

class SQLDataLoader:

    engine = create_engine("postgresql://admin:10293847@localhost:5432/datapura")
    
    @staticmethod
    def ingest(file_path: str):
        table_name = os.path.splitext(os.path.basename(file_path))[0].lower()
        ext = os.path.splitext(file_path)[1].lower()

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
            # Normalize column names for SQL BEFORE writing the table
            
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(r'[^a-z0-9_]+', '_', regex=True)
            )

            df.to_sql(table_name, SQLDataLoader.engine, if_exists="replace", index=False)

            return pd.read_sql(f"SELECT * FROM {table_name}", SQLDataLoader.engine)

        elif ext == ".sql":
            with open(file_path, "r", encoding="utf-8") as f:
                sql_text = f.read()

            # The BALANCED Solution: Split by semicolon and filter out empty lines/comments.
            # This is simple and works for most clean SQL scripts (if strings don't contain ';').
            statements = [
                s.strip()
                for s in sql_text.split(";")
                if s.strip() and not s.strip().startswith(("--", "/*"))
            ]

            if not statements:
                # Handle case where the file is empty or only comments
                return pd.DataFrame() 

            # Execute statements in a single transaction
            try:
                with SQLDataLoader.engine.begin() as conn:
                    for stmt in statements:
                        conn.execute(text(stmt))
            except Exception as e:
                # Reraise with context
                raise RuntimeError(f"Failed executing SQL file {file_path}. Details: {e}") from e

            # Fallback: After execution, attempt to read the table named after the file.
            # This assumes the SQL script created a table matching the file name (e.g., cars.sql -> 'cars' table)
            try:
                return pd.read_sql(f"SELECT * FROM {table_name}", SQLDataLoader.engine)
            except Exception:
                # If the table isn't readable, return an empty DataFrame or raise an error.
                print(f"Warning: SQL script executed, but could not read table '{table_name}'.")
                return pd.DataFrame()

        else:
            raise ValueError(f"Unsupported format: {ext}")