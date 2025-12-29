import pandas as pd
from typing import Union

class Insights:
    
    def __init__(self):
        pass
    
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
        
        print("✅ Distribution Rates Available")
        return pd.DataFrame(result).T
    
    def missing_rows(self, df: pd.DataFrame,
                     drop_threshold: float = None,
                     axis: str = None,
                     inplace: bool = False) -> pd.DataFrame:
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
        return 
    
    
    def unique_items_list(self, 
                          df: pd.DataFrame, 
                          columns: Union[list, str], 
                          count: bool = False, 
                          error_skip: bool = False):
        
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
        
        print("✅ Unique Items List Available")
        print(result)
        return result

    
    def min_max_finder(self,
                       df: pd.DataFrame,
                       columns: Union[list, str], 
                       error_skip: bool = False) -> list:
          
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
        print("✅ Min/Max Finding Complete")