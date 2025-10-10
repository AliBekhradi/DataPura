import os
import tempfile
import pytest
from unittest.mock import patch
import pandas as pd

class TestFrequencyEncoder:

    def test_encode_single_column(self, prep, freq_encoding_df):
        df = freq_encoding_df.copy()
        result = prep.FrequencyEncoder(df, columns="department", mapping_return=False)

        # Check if frequency-encoded column exists
        assert "department_freq" in result.columns

        # Check frequencies are between 0 and 1
        assert result["department"].min() >= 0.0
        assert result["department"].max() <= 1.0

    def test_encode_multiple_columns(self, prep, freq_encoding_df):
        df = freq_encoding_df.copy()
        result = prep.FrequencyEncoder(df, columns=["department", "region"], mapping_return=False)

        assert "department_freq" in result.columns
        assert "region_freq" in result.columns

        assert result["department"].max() <= 1.0
        assert result["department"].min() >= 0.0
        assert result["region"].max() <= 1.0
        assert result["region"].min() >= 0.0
    
    """QUESTIONS ABOUT THIS PART"""
    def test_encode_invalid_column(self, prep, freq_encoding_df):
        df = freq_encoding_df.copy()
        with pytest.raises(KeyError):
            prep.FrequencyEncoder(df, columns="nonexistent_column")

    def test_mapping_is_saved(self, prep, freq_encoding_df):
        """QUESTIONS ABOUT THIS PART"""
        df = freq_encoding_df.copy() 
        """QUESTIONS ABOUT THIS PART"""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:

            # Patch MappingManager.save_mapping to redirect to temp_dir
            with patch("main.core.MappingManager.save_mapping") as mock_save:
                
                def fake_save(col, mapping, format="joblib"):
                    ext = "joblib"  # Frequency mappings are dicts, assume joblib
                    path = os.path.join(temp_dir, f"{col}_freq_mapping.{ext}")
                    import joblib
                    joblib.dump(mapping, path)

                mock_save.side_effect = fake_save

                result = prep.FrequencyEncoder(df, columns="region", mapping_return=True)

                assert "region_freq" in result.columns
                expected_path = os.path.join(temp_dir, "region_freq_mapping.joblib")
                assert os.path.exists(expected_path)

class TestTargetEncoder:

    def test_target_encode_single_column(self, prep, target_encoding_df):
        df = target_encoding_df.copy()
        result = prep.TargetEncoder(df, columns="department", target="performance_score", mapping_return=False)

        assert "department" in result.columns
        assert pd.api.types.is_numeric_dtype(result["department"])

    def test_target_encode_multiple_columns(self, prep, target_encoding_df):
        df = target_encoding_df.copy()
        result = prep.TargetEncoder(df, columns=["department", "region"], target="performance_score", mapping_return=False)

        assert "department" in result.columns
        assert "region" in result.columns
        assert pd.api.types.is_numeric_dtype(result["department"])
        assert pd.api.types.is_numeric_dtype(result["region"])

    def test_target_encode_invalid_column(self, prep, target_encoding_df):
        df = target_encoding_df.copy()
        with pytest.raises(KeyError):
            prep.TargetEncoder(df, columns="nonexistent_column", target="performance_score")

    def test_target_column_coercion(self, prep, target_encoding_df):
        df = target_encoding_df.copy()
        df["performance_score"] = df["performance_score"].astype(str)  # Make target non-numeric

        result = prep.TargetEncoder(df, columns="department", target="performance_score", mapping_return=False)
        assert pd.api.types.is_numeric_dtype(result["department"])

    def test_mapping_is_saved(self, prep, target_encoding_df):
        df = target_encoding_df.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("main.core.MappingManager.save_mapping") as mock_save:

                def fake_save(col, encoder_obj, format="joblib"):
                    path = os.path.join(temp_dir, f"{col}_target_encoder.joblib")
                    import joblib
                    joblib.dump(encoder_obj, path)

                mock_save.side_effect = fake_save

                result = prep.TargetEncoder(df, columns="region", target="performance_score", mapping_return=True)

                expected_path = os.path.join(temp_dir, "region_target_encoder.joblib")
                assert os.path.exists(expected_path)
                
class TestOneHotEncoder:
    
    def test_onehot_encode_color_column(self, prep, onehot_encoding_df):
        df = onehot_encoding_df.copy()
        result_df = prep.OneHotEncoder(df, columns="color", mapping_return=False)

        # Ensure 'color' column is removed
        assert "color" not in result_df.columns

        # Ensure expected one-hot columns are present
        expected_columns = {"color_Blue", "color_Green", "color_Red"}
        assert expected_columns.issubset(set(result_df.columns))

        # Ensure row count remains the same
        assert result_df.shape[0] == df.shape[0]

        # Check original non-encoded data is preserved
        assert "size" in result_df.columns
        assert "price" in result_df.columns

    def test_onehot_encode_multiple_columns(self, prep, onehot_encoding_df):
        df = onehot_encoding_df.copy()
        result_df = prep.OneHotEncoder(df, columns=["color", "size"], mapping_return=False)

        # Ensure original columns are removed
        assert "color" not in result_df.columns
        assert "size" not in result_df.columns

        # Ensure one-hot columns are added
        expected_color = {"color_Blue", "color_Green", "color_Red"}
        expected_size = {"size_S", "size_M", "size_L", "size_XL"}
        assert expected_color.issubset(set(result_df.columns))
        assert expected_size.issubset(set(result_df.columns))

        # Row count check
        assert result_df.shape[0] == df.shape[0]

        # Price column must still be there
        assert "price" in result_df.columns

    def test_missing_column_raises_keyerror(self, prep, onehot_encoding_df):
        df = onehot_encoding_df.copy()
        with pytest.raises(KeyError, match=r"⚠️ Column 'nonexistent' does not exist in the DataFrame\."):
            prep.OneHotEncoder(df, columns="nonexistent", mapping_return=False)