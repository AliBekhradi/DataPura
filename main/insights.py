from typing import Union, List, Dict
from sqlalchemy import text

class Insights:
    
    def __init__(self):
        pass
    
    def distribution_rates(
        self,
        engine,
        table: str,
        columns: Union[str, List[str]],
        display: bool = True,
        schema: str = "public"
    ) -> Dict:

        if isinstance(columns, str):
            columns = [columns]

        results = {}

        with engine.begin() as conn:
            for col in columns:
                query = text(f"""
                    SELECT
                        COUNT(*) AS total_rows,
                        COUNT(*) FILTER (WHERE {col} IS NULL) AS null_count,
                        COUNT(*) FILTER (WHERE {col} = 0) AS zero_count
                    FROM {schema}.{table};
                """)

                row = conn.execute(query).mappings().one()

                total = row["total_rows"]
                nulls = row["null_count"]
                zeros = row["zero_count"]
                safe = total - nulls - zeros

                results[col] = {
                    "safe_count": safe,
                    "safe_prc": f"% {round((safe / total) * 100, 2) if total else 0}",
                    "zero_count": zeros,
                    "zero_prc": f"% {round((zeros / total) * 100, 2) if total else 0}",
                    "null_count": nulls,
                    "null_prc": f"% {round((nulls / total) * 100, 2) if total else 0}"
                }

        print("✅ Distribution Rates Available")
        if display:
            print("=" * 50)
            print("📊 Distribution Rates")
            print("=" * 50)
            for col, stats in results.items():
                print(f"\nColumn: {col}")
                for k, v in stats.items():
                    print(f"  {k.replace('_', ' ').title()}: {v}")

        return {
            "operation": "distribution_rates",
            "table": f"{schema}.{table}",
            "results": results,
            "status": "success"
        }
    
    def min_max_finder(
        self,
        engine,
        table,
        columns: Union[List[str], str],
        display: bool = True,
        schema = "public",
        ) -> Dict:
          
        if isinstance(columns, str):
            columns = [columns]
            
        results = {}
        
        with engine.begin() as conn:
            for col in columns:
                query = text(f"""
                SELECT
                    MIN({col}) AS min_value,
                    MAX({col}) AS max_value
                FROM {schema}.{table};
            """)
                
            row = conn.execute(query).mappings().one()

            results[col] = {
                "min": row["min_value"],
                "max": row["max_value"]
            }
        
        print("✅ Min/Max Finding Complete")
        
        if display:
            print("=" * 40)
            print("📈 Min / Max Finder")
            print("=" * 40)
            for col, stats in results.items():
                print(f"{col}:")
                print(f"  Min: {stats['min']}")
                print(f"  Max: {stats['max']}")
                print()

        return {
        "operation": "min_max_finder",
        "table": f"{schema}.{table}",
        "results": results,
        "status": "success"
    }
    
    def missing_rows(
        self,
        engine,
        table: str,
        column: Union[str, List[str]],
        schema: str = "public",
        drop_threshold: float = None,
        axis: str = None,
        display: bool = True
    ) -> Dict:

        with engine.begin() as conn:

            # Column-level missing stats
            query = text(f"""
                SELECT
                    column_name,
                    COUNT(*) FILTER (WHERE {table}.{column} IS NULL)::float
                    / COUNT(*) AS missing_ratio
                FROM information_schema.columns
                JOIN {schema}.{table} ON true
                WHERE table_schema = :schema
                AND table_name = :table
                GROUP BY column_name;
            """)

            rows = conn.execute(
                query, {"schema": schema, "table": table}
            ).mappings().all()

        missing_report = {
            row["column_name"]: round(row["missing_ratio"] * 100, 2)
            for row in rows
        }


        if display:
            print("=" * 50)
            print("📊 Missing Data Report")
            print("=" * 50)

            any_missing = False

            for col, pct in missing_report.items():
                if pct > 0:
                    any_missing = True
                    print(f"{col}: {pct}% missing")

            if not any_missing:
                print("✅ No missing values detected in any of the specified columns.")

        return {
            "operation": "missing_rows",
            "table": f"{schema}.{table}",
            "missing_report": missing_report,
            "drop_threshold": drop_threshold,
            "axis": axis,
            "status": "success"
        }

    
    def unique_items_list(
        self,
        engine,
        table: str,
        columns: Union[str, List[str]],
        schema: str = "public",
        display: bool = True
    ) -> Dict:

        if isinstance(columns, str):
            columns = [columns]

        HIGH_CARDINALITY_THRESHOLD = 0.3
        results = {}

        with engine.begin() as conn:

            # Total row count (once)
            total_rows = conn.execute(
                text(f"SELECT COUNT(*) FROM {schema}.{table};")
            ).scalar()

            for col in columns:

                # Unique value count
                unique_count = conn.execute(
                    text(f"""
                        SELECT COUNT(DISTINCT {col})
                        FROM {schema}.{table};
                    """)
                ).scalar()

                # Distribution per value
                rows = conn.execute(
                    text(f"""
                        SELECT
                            {col} AS value,
                            COUNT(*) AS count,
                            ROUND(
                                (
                                    COUNT(*)::numeric
                                    / SUM(COUNT(*)) OVER ()
                                    * 100
                                ),
                                2
                            ) AS percentage
                        FROM {schema}.{table}
                        GROUP BY {col}
                        ORDER BY count DESC;
                    """)
                ).mappings().all()

                cardinality_ratio = (
                    unique_count / total_rows if total_rows else 0
                )

                results[col] = {
                    "unique_count": unique_count,
                    "total_rows": total_rows,
                    "cardinality_ratio": cardinality_ratio,
                    "distribution": rows
                }

        # Human-facing output
        if display:
            print("=" * 50)
            print("📊 Value Distribution")
            print("=" * 50)

            for col, data in results.items():
                print(f"\nColumn: {col}")
                print(f"Unique values: {data['unique_count']}")

                if data["cardinality_ratio"] > HIGH_CARDINALITY_THRESHOLD:
                    print(
                        f"⚠️  Column '{col}' has very high cardinality "
                        f"({data['unique_count']} unique / "
                        f"{data['total_rows']} rows). "
                        f"Distribution skipped."
                    )
                else:
                    for row in data["distribution"]:
                        print(
                            f"  {row['value']}: "
                            f"{row['count']} "
                            f"({row['percentage']}%)"
                        )

        return {
            "operation": "unique_items_list",
            "table": f"{schema}.{table}",
            "results": results,
            "status": "success"
        }