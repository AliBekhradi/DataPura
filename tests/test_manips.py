import os
from sklearn.preprocessing import MinMaxScaler
import pytest
import pandas as pd
import tempfile
import shutil
from unittest.mock import patch

class TestImputator:

    def test_mean_imputation(self, prep, imputation_df):
        result = prep.imputator(imputation_df.copy(), "age", "mean")
        assert result["age"].isna().sum() == 0

    def test_median_imputation(self, prep, imputation_df):
        result = prep.imputator(imputation_df.copy(), "income", "median")
        assert result["income"].isna().sum() == 0

    def test_mode_imputation(self, prep):
        df = pd.DataFrame({
            "group": [1, 2, 2, None, 3, None]
        })
        result = prep.imputator(df, "group", "mode")
        assert result["group"].isna().sum() == 0
        assert result["group"].mode()[0] == 2

    def test_interpolation_imputation(self, prep):
        df = pd.DataFrame({
            "val": [1, None, None, 4, 5]
        })
        result = prep.imputator(df, "val", "interpolate")
        assert result["val"].isna().sum() == 0

    def test_multiple_columns_multiple_modes(self, prep, imputation_df):
        result = prep.imputator(imputation_df.copy(), ["age", "income"], ["mean", "median"])
        assert result["age"].isna().sum() == 0
        assert result["income"].isna().sum() == 0

    def test_invalid_column_raises_keyerror(self, prep, imputation_df):
        with pytest.raises(KeyError):
            prep.imputator(imputation_df.copy(), ["invalid_col"], "mean")

    def test_invalid_mode_raises_valueerror(self, prep, imputation_df):
        with pytest.raises(ValueError):
            prep.imputator(imputation_df.copy(), ["age"], "bad_mode")

    def test_mismatched_list_lengths_raises_valueerror(self, prep, imputation_df):
        with pytest.raises(ValueError):
            prep.imputator(imputation_df.copy(), ["age", "income"], ["mean"])
            
class TestNormalization:

    def test_normalize_single_column(self, prep, normalization_df):
        df = normalization_df.copy()
        result = prep.normalize(df, columns="salary", save_scaler=False)
        
        assert result["salary"].min() == pytest.approx(0.0)
        assert result["salary"].max() == pytest.approx(1.0)

    def test_normalize_multiple_columns(self, prep, normalization_df):
        df = normalization_df.copy()
        result = prep.normalize(df, columns=["salary", "bonus"], save_scaler=False)

        assert result["salary"].min() == pytest.approx(0.0)
        assert result["salary"].max() == pytest.approx(1.0)
        assert result["bonus"].min() == pytest.approx(0.0)
        assert result["bonus"].max() == pytest.approx(1.0)

    def test_normalize_invalid_column(self, prep, normalization_df):
        df = normalization_df.copy()
        with pytest.raises(KeyError):
            prep.normalize(df, columns=["nonexistent_column"])

    def test_scaler_is_saved(self, prep, normalization_df):
        df = normalization_df.copy()
        col_name = "salary"
        scaler_path = f"scalers/{col_name}_scaler.pkl"

        # Clean up before test
        if os.path.exists(scaler_path):
            os.remove(scaler_path)

        prep.normalize(df, columns=col_name, save_scaler=True)

        assert os.path.exists(scaler_path)

        # Clean up after test
        os.remove(scaler_path)

    def test_mapping_return_does_not_crash(self ,prep, normalization_df):
        df = normalization_df.copy()

        # Create a temporary directory for saving mappings
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Patch MappingManager's save_mapping method to write to temp_dir instead
            with patch("main.core.MappingManager.save_mapping") as mock_save:
                
                def fake_save(col, scaler, format="joblib"):
                    file_path = os.path.join(temp_dir, f"{col}_mapping.{format}")
                    if format == "json":
                        with open(file_path, "w") as f:
                            f.write("{}")  # dummy json content
                    else:
                        import joblib
                        joblib.dump(scaler, file_path)
                        
                mock_save.side_effect = fake_save

                result = prep.normalize(df, columns="bonus", mapping_return=True, save_scaler=False)

                # Check that the column was normalized
                assert "bonus" in result.columns
                assert result["bonus"].min() == 0.0
                assert result["bonus"].max() == 1.0

                # Check that a mapping file was saved to the temp dir
                expected_path = os.path.join(temp_dir, "bonus_mapping.joblib")
                assert os.path.exists(expected_path)
        
class TestSaveDataFrame:

    def test_save_as_csv(self, prep, save_dataframe_df):
        path = "test_output.csv"
        prep.save_dataframe(save_dataframe_df, output_path=path, file_format="csv")
        assert os.path.exists(path)
        os.remove(path)

    def test_save_as_json(self, prep, save_dataframe_df):
        path = "test_output.json"
        prep.save_dataframe(save_dataframe_df, output_path=path, file_format="json")
        assert os.path.exists(path)
        os.remove(path)

    def test_save_as_jsonl(self, prep, save_dataframe_df):
        path = "test_output.jsonl"
        prep.save_dataframe(save_dataframe_df, output_path=path, file_format="jsonl")
        assert os.path.exists(path)
        os.remove(path)

    def test_save_as_excel(self, prep, save_dataframe_df):
        path = "test_output.xlsx"
        prep.save_dataframe(save_dataframe_df, output_path=path, file_format="excel")
        assert os.path.exists(path)
        os.remove(path)

    def test_invalid_format_does_not_crash(self, prep, save_dataframe_df):
        # This just ensures the method doesn't raise an exception
        result = prep.save_dataframe(save_dataframe_df, output_path="test_output.txt", file_format="txt")
        assert isinstance(result, pd.DataFrame)