import pytest
import pandas as pd

class TestUniqueItemsList:

    def test_single_column_city(self, prep, unique_items_df):
        result = prep.unique_items_list(unique_items_df, 'city')
        assert 'city' in result
        assert result['city'] == ['', 'Isfahan', 'Mashhad', 'Tehran', 'tehran']

    def test_multiple_columns(self, prep, unique_items_df):
        result = prep.unique_items_list(unique_items_df, ['city', 'gender'])
        assert 'city' in result and 'gender' in result
        assert sorted(result['gender']) == ['', 'FEMALE', 'Female', 'Male', 'male']

    def test_missing_column(self, prep, unique_items_df):
        with pytest.raises(KeyError):
            prep.unique_items_list(unique_items_df, 'country')

    def test_nan_and_empty_handling(self, prep, unique_items_df):
        result = prep.unique_items_list(unique_items_df, 'joined')
        assert '' in result['joined']
        assert None not in result['joined']  # NaNs should be dropped

    def test_count_true_prints(self, prep, unique_items_df, capsys):
        result = prep.unique_items_list(unique_items_df, ['city'], count=True)
        captured = capsys.readouterr()
        assert "city:" in captured.out
        assert isinstance(result, dict)

    def test_string_coercion_and_strip(self, prep):
        df = pd.DataFrame({
            'mixed': [' 1 ', 1, '1', ' 2', '2 ', None]
        })
        result = prep.unique_items_list(df, 'mixed')
        assert result['mixed'] == ['1', '2']

    def test_sorted_result(self, prep):
        df = pd.DataFrame({
            'letters': ['b', 'a', 'C', 'c', 'A']
        })
        result = prep.unique_items_list(df, 'letters')
        assert result['letters'] == ['A', 'C', 'a', 'b', 'c']  # Default sorted by string ASCII
        
class TestMinMaxFinder:
    
    def test_min_max_finder_basic(self, prep, minmax_df, capsys):
        prep.min_max_finder(minmax_df, ['price', 'discount'])
        captured = capsys.readouterr().out
        assert "price:" in captured
        assert "min" in captured and "max" in captured
        assert "100" in captured and "300" in captured
        assert "discount:" in captured
        assert "0" in captured and "15" in captured

    def test_min_max_finder_single_column(self, prep, minmax_df, capsys):
        prep.min_max_finder(minmax_df, 'quantity')
        out = capsys.readouterr().out
        assert "quantity:" in out
        assert "1" in out and "5" in out

    def test_min_max_finder_missing_column(self, prep, minmax_df):
        with pytest.raises(KeyError) as e:
            prep.min_max_finder(minmax_df, ['notacolumn'])
        assert "notacolumn" in str(e.value)

class TestMissingRows:
    
    def test_no_drop_operation(self, prep, missingrows_df):
        # Just run it without drop_threshold or axis to check basic print + report
        result = prep.missing_rows(missingrows_df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_drop_rows_inplace(self, prep, missingrows_df):
        df_copy = missingrows_df.copy()
        original_len = len(df_copy)
        prep.missing_rows(df_copy, drop_threshold=0.5, axis='rows', inplace=True)
        assert len(df_copy) <= original_len  # rows may be dropped

    def test_drop_columns_inplace(self, prep, missingrows_df):
        df_copy = missingrows_df.copy()
        original_cols = set(df_copy.columns)
        prep.missing_rows(df_copy, drop_threshold=0.5, axis='columns', inplace=True)
        assert set(df_copy.columns).issubset(original_cols)

    def test_drop_rows_not_inplace(self, prep, missingrows_df):
        df_copy = missingrows_df.copy()
        result = prep.missing_rows(df_copy, drop_threshold=0.5, axis='rows', inplace=False)
        assert result is not df_copy
        assert len(result) <= len(df_copy)
        assert df_copy.equals(missingrows_df)  # unchanged

    def test_drop_columns_not_inplace(self, prep, missingrows_df):
        df_copy = missingrows_df.copy()
        result = prep.missing_rows(df_copy, drop_threshold=0.5, axis='columns', inplace=False)
        assert result is not df_copy
        assert df_copy.equals(missingrows_df)  # unchanged

    def test_invalid_threshold_raises(self, prep, missingrows_df):
        with pytest.raises(ValueError):
            prep.missing_rows(missingrows_df, drop_threshold=1.5, axis='rows')

    def test_invalid_axis_raises(self, prep, missingrows_df):
        with pytest.raises(ValueError):
            prep.missing_rows(missingrows_df, drop_threshold=0.5, axis='diagonal')