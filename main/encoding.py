from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import pandas as pd
from typing import Union
from mapping import MappingManager

class Encoders:
    
    def __init__(self):
        pass
    
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

        print("✅ One Hot Encoding Complete")
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

        print("✅ Target Encoding Complete")
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

        print("✅ Frequency Encoding Complete")
        return freq_df