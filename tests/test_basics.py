import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal

class TestRowSampling:
    
    def test_returns_dataframe(self, prep, rows_sampling_df):
        result = prep.rows_sampling(rows_sampling_df, 10)
        assert isinstance(result, pd.DataFrame)

    def test_row_count(self, prep, rows_sampling_df):
        result = prep.rows_sampling(rows_sampling_df, 10)
        assert len(result) == 10

    def test_column_integrity(self, prep, rows_sampling_df):
        result = prep.rows_sampling(rows_sampling_df, 10)
        assert set(result.columns) == set(rows_sampling_df.columns)

    def test_deterministic_sampling(self, prep, rows_sampling_df):
        r1 = prep.rows_sampling(rows_sampling_df, 10)
        r2 = prep.rows_sampling(rows_sampling_df, 10)
        pd.testing.assert_frame_equal(r1, r2)

    def test_invalid_n_type(self, prep, rows_sampling_df):
        with pytest.raises(TypeError):
            prep.rows_sampling(rows_sampling_df, "ten")

    def test_accepts_csv_path(self, tmp_path, prep, rows_sampling_df):
        csv_path = tmp_path / "data.csv"
        rows_sampling_df.to_csv(csv_path, index=False)
        result = prep.rows_sampling(str(csv_path), 5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert set(result.columns) == set(rows_sampling_df.columns)

class TestColumnsRename:

    def test_columns_rename_strict(self, prep, column_rename_df):
        result = prep.columns_rename(
            column_rename_df,
            columns="first_name",
            rename_to="name",
            error_skip=False
        )

        assert "name" in result.columns
        assert "first_name" not in result.columns

    def test_columns_rename_multiple_columns(self, prep, column_rename_df):
        result = prep.columns_rename(
            column_rename_df,
            columns=["first_name", "last_name"],
            rename_to=["fname", "lname"],
            error_skip = False
        )

        assert "fname" in result.columns
        assert "lname" in result.columns
        assert "first_name" not in result.columns
        assert "last_name" not in result.columns

    def test_columns_rename_nonexistent_strict(self, prep, column_rename_df):
        with pytest.raises(KeyError):
            prep.columns_rename(
                column_rename_df,
                columns="nonexistent",
                rename_to="whatever",
                error_skip=False
            )

    def test_columns_rename_nonexistent_nonstrict(self, prep, column_rename_df):
        result = prep.columns_rename(
            column_rename_df,
            columns=["first_name", "nonexistent"],
            rename_to=["fname", "missing"],
            error_skip=True
        )

        assert "fname" in result.columns
        assert "first_name" not in result.columns
        assert "nonexistent" not in result.columns  # It was never there
        assert "missing" not in result.columns      # Shouldn't exist either

class TestColumnsDrop:

    def test_columns_drop_single_column(self, prep, column_drop_df):
        result = prep.columns_drop(column_drop_df, "city")
        assert "city" not in result.columns
        assert result.shape[1] == 3

    def test_columns_drop_multiple_columns(self, prep, column_drop_df):
        result = prep.columns_drop(column_drop_df, ["city", "income"])
        assert "city" not in result.columns
        assert "income" not in result.columns
        assert result.shape[1] == 2

    def test_columns_drop_nonexistent_column(self, prep, column_drop_df):
        with pytest.raises(KeyError):
            prep.columns_drop(column_drop_df, ["gender"])
            
class TestDupRowRemover:

    def test_remove_all_duplicates(self, prep, rows_duplicate_remover_df):
        result = prep.dup_row_remover(rows_duplicate_remover_df, columns=["name", "city", "age"], keep=False)
        # Only unique rows should remain
        assert result.shape[0] == 1
        expected_names = {"Reza"}
        assert set(result["name"]) == expected_names

    def test_remove_duplicates_keep_first(self, prep, rows_duplicate_remover_df):
        result = prep.dup_row_remover(rows_duplicate_remover_df, columns=["name"], keep="first")
        assert "Ali" in result["name"].values
        assert result["name"].tolist().count("Ali") == 1

    def test_remove_duplicates_keep_last(self, prep, rows_duplicate_remover_df):
        result = prep.dup_row_remover(rows_duplicate_remover_df, columns=["name"], keep="last")
        assert "Ali" in result["name"].values
        # Get the index of the last occurrence of Ali — this works now
        ali_index = result[result["name"] == "Ali"].index[0]
        assert ali_index > 0
        # Make sure only one "Ali" remains
        assert result["name"].tolist().count("Ali") == 1

    def test_raises_key_error_on_invalid_column(self, prep, rows_duplicate_remover_df):
        with pytest.raises(KeyError):
            prep.dup_row_remover(rows_duplicate_remover_df, columns="invalid_column")

    def test_raises_value_error_on_invalid_keep(self, prep, rows_duplicate_remover_df):
        with pytest.raises(ValueError):
            prep.dup_row_remover(rows_duplicate_remover_df, columns="name", keep="invalid_option")
            
class TestCaseConversion:
    
    def test_lowercase(self, prep, convert_case_df):
        lower_df = prep.convert_case(convert_case_df.copy(), ['name', 'country'], 'lower')
        assert all(lower_df['name'] == ['john doe', 'jane doe', 'alice mcdonald'])
        assert all(lower_df['country'] == ['usa', 'germany', 'uae'])

    def test_uppercase(self, prep, convert_case_df):
        upper_df = prep.convert_case(convert_case_df.copy(), ['name', 'country'], 'upper')
        assert all(upper_df['name'] == ['JOHN DOE', 'JANE DOE', 'ALICE MCDONALD'])
        assert all(upper_df['country'] == ['USA', 'GERMANY', 'UAE'])

    def test_title(self, prep, convert_case_df):
        title_df = prep.convert_case(convert_case_df.copy(), ['name', 'country'], 'title')
        assert all(title_df['name'] == ['John Doe', 'Jane Doe', 'Alice Mcdonald'])
        assert all(title_df['country'] == ['Usa', 'Germany', 'Uae'])

    def test_capitalization(self, prep, convert_case_df):
        capitalized_df = prep.convert_case(convert_case_df.copy(), ['name', 'country'], 'capitalize')
        assert all(capitalized_df['name'] == ['John Doe', 'Jane Doe', 'Alice Mcdonald'])
        assert all(capitalized_df['country'] == ['Usa', 'Germany', 'Uae'])

    def test_invalid_mode(self, prep, convert_case_df):
        with pytest.raises(ValueError):
            prep.convert_case(convert_case_df.copy(), ['name'], 'snake_case')

    def test_invalid_column(self, prep, convert_case_df):
        with pytest.raises(KeyError):
            prep.convert_case(convert_case_df.copy(), ['invalid_column'], 'lower')
            
class TestRemoveWhitespace:
    
    def test_remove_whitespace_single_column(self, prep, remove_whitespace_df):
        result = prep.remove_whitespace(remove_whitespace_df, "city")
        
        expected_cities = ["Tehran", "Los Angeles", "Dubai", "Venice", "Istanbul", "London"]
        
        assert result["city"].tolist() == expected_cities
        
    def test_remove_whitespace_multiple_columns(self, prep, remove_whitespace_df):
        result = prep.remove_whitespace(remove_whitespace_df.copy(), columns=["name", "city", "country"])

        expected_names = ["Alex Bolton", "Sara Stacy", "Peter Parker", "Al Pacino", "Sara", "Ali"]
        expected_cities = ["Tehran", "Los Angeles", "Dubai", "Venice", "Istanbul", "London"]
        expected_countries = ["Iran", "USA", "UAE", "Italy", "Turkey", "UK"]

        assert result["name"].tolist() == expected_names
        assert result["city"].tolist() == expected_cities
        assert result["country"].tolist() == expected_countries
    
    def test_invalid_column(self, prep, remove_whitespace_df):
        with pytest.raises(KeyError):
            prep.remove_whitespace(remove_whitespace_df.copy(), ['invalid_column'])
            
    def test_returns_dataframe(self, prep, remove_whitespace_df):
        result = prep.remove_whitespace(remove_whitespace_df, "city")
        assert isinstance(result, pd.DataFrame)
        
        
class TestStandardizeDates:
    
    def test_standardize_dates(self, prep, standardize_date_df):
            cleaned = prep.standardize_dates(standardize_date_df.copy(), columns=["start_date"])

            expected = [
                "2023-01-01",
                "2023-01-02",
                "2023-03-01",
                pd.NaT,
                "2023-07-13",
                "2023-01-01",
                "2025-08-05",
                "invalid-date"  # Left as-is
            ]

            result = cleaned["start_date"].tolist()

            # Compare parsed dates (convert to string if necessary)
            for i, val in enumerate(expected):
                if pd.isna(val):
                    assert pd.isna(result[i])
                else:
                    assert result[i] == val
    
    def test_invalid_column(self, prep, standardize_date_df):
        with pytest.raises(KeyError):
            prep.standardize_dates(standardize_date_df.copy(), ['invalid_column'])                
    
    def test_returns_dataframe(self, prep, standardize_date_df):
        result = prep.standardize_dates(standardize_date_df, "start_date")
        assert isinstance(result, pd.DataFrame)
        
class TestFormatNumbers:
    
    def test_format_numbers_basic(self, prep, format_numbers_df):
        
        df = prep.format_numbers(format_numbers_df.copy(), columns=['price', 'revenue'], decimal_places=1)

        expected_price = pd.Series([1200.5, 3451.0, -720.3, np.nan, np.nan], name='price')
        expected_revenue = pd.Series([1000.0, 2500.5, np.nan, 3000.0, 4000.0], name='revenue')

        assert_series_equal(df['price'], expected_price)
        assert_series_equal(df['revenue'], expected_revenue)
    
    def test_format_numbers_drop_invalid(self, prep, format_numbers_df):
        
        df = prep.format_numbers(format_numbers_df.copy(), columns='price', drop_invalid=True, decimal_places=2)
        
        # Should only keep rows where 'price' was convertible
        expected_prices = [pytest.approx(1200.5), pytest.approx(3450.99), pytest.approx(-720.3)]
        
        assert df['price'].tolist() == expected_prices
        assert len(df) == 3  # Only 3 valid rows


    def test_format_numbers_missing_column_continue(self, prep, format_numbers_df, capsys):
        df = prep.format_numbers(format_numbers_df.copy(), columns=['nonexistent_column'], error_skip=True)
        captured = capsys.readouterr()
        assert "does not exist in the DataFrame" in captured.out
        assert isinstance(df, pd.DataFrame)  # Still returns a DataFrame


    def test_invalid_column(self, prep, format_numbers_df):
        with pytest.raises(KeyError):
            prep.format_numbers(format_numbers_df.copy(), ['invalid_column'], error_skip = False)                
    
    def test_returns_dataframe(self, prep, format_numbers_df):
        result = prep.format_numbers(format_numbers_df, columns = ["price", "revenue"])
        assert isinstance(result, pd.DataFrame)
        
class TestRemoveIrrelavantCharacter:
    
    def test_remove_irrelevant_characters_single_column(self, prep, remove_irrelavant_characters_df):
        df = prep.remove_irrelevant_characters(remove_irrelavant_characters_df.copy(), columns="title")

        expected_title = pd.Series(
            ["Hello World!", "Extra spaces here", "Special characters"], name='title'
        )

        assert_series_equal(df['title'], expected_title)
        # Make sure "notes" column stays unchanged
        assert df['notes'].equals(remove_irrelavant_characters_df['notes'])

    def test_remove_irrelevant_characters_multiple_columns(self, prep, remove_irrelavant_characters_df):
        df = prep.remove_irrelevant_characters(remove_irrelavant_characters_df.copy(), columns=["title", "notes"])

        expected_title = pd.Series(["Hello World!", "Extra spaces here", "Special characters"], name='title')
        expected_notes = pd.Series(["Some text", None, "Clean already"], name='notes')

        assert_series_equal(df['title'], expected_title)
        assert_series_equal(df['notes'], expected_notes)

    def test_missing_column_continue(self, prep, remove_irrelavant_characters_df, capsys):
        df = prep.remove_irrelevant_characters(remove_irrelavant_characters_df.copy(), columns=['nonexistent_column'], error_skip=True)
        captured = capsys.readouterr()

        assert "does not exist in the DataFrame" in captured.out
        assert isinstance(df, pd.DataFrame)

    def test_missing_column_raise(self, prep, remove_irrelavant_characters_df):
        with pytest.raises(KeyError):
            prep.remove_irrelevant_characters(remove_irrelavant_characters_df.copy(), columns=['invalid_column'], error_skip=False)

    def test_returns_dataframe(self, prep, remove_irrelavant_characters_df):
        result = prep.remove_irrelevant_characters(remove_irrelavant_characters_df.copy(), columns=['title'])
        assert isinstance(result, pd.DataFrame)