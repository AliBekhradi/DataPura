import pandas as pd
from typing import Union
import re

class stringsprocessor:

    def __init__(self):
        pass
    
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

        print("✅ Removed The Requested Strings From The Defined Columns")
        return string_removed_df
    
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
                print(f"✅ {col} Succesfully Converted To Lowercase")
            elif mode == 'upper':
                case_convert_df[col] = case_convert_df[col].str.upper()
                print(f"✅ {col} Succesfully Converted To Uppercase")
            elif mode == 'title':
                case_convert_df[col] = case_convert_df[col].str.title()
                print(f"✅ {col} Succesfully Converted To Title")
            elif mode == 'capitalize':
                case_convert_df[col] = case_convert_df[col].apply(lambda x: ' '.join([word.capitalize() for word in x.split()]))
                print(f"✅ {col} Succesfully Capitalized")
                
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
        
        print("✅ Whitespace Removed")
                    
        return whitespace_df
    
    def remove_irrelevant_characters(self, df: pd.DataFrame, columns: Union[list, str], error_skip: bool = False):
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

        print("✅ Irrelavant Characters Removed")
        return irrelevant_df