# DataPura
GUIDE:

A. MappingManager
    1. save_mapping
    2. load_ mapping
    3. apply_mapping
    
B. Preprocessor
    B1. Row & Value Handling
        1. rows_sampling
        2. value_rows_remover
        3. imputator
        4. distribution_rates
    B2. String Cleaning
        1. batch_string_remover
        2. convert_case
        3. remove_whitespace
        4. remove_irrelavant characters
    B3. Dates & Numbers
        1. standardize_dates
        2. format_numbers
    B4. Category Fixing & Encoding
        1. normalize_categories
        2. FrequencyEncoder
        3. OneHotEncoder
        4. TargetEncoder
        5. normalize
    B5. Column Editing
        1. column_rename
        2. column_drop
        3. value_remover
        4. dup_row_remover
    B6. Diagnostics & Helpers
        1. missing_rows
        2. unique_items_list
        3. min_max_finder
    B7. Saving Data
        1. save_dataframe


---------- A. MappingManager ----------

1. save_mapping(column_name, mapping, format="joblib")

What it does: Saves a mapping (dict, encoder, scaler, etc.) to disk.

When to use:
 - When you need to reuse the same scaler/encoder/mapping on new data later.

2. load_mapping(column_name, format="joblib")

What it does: Loads a previously saved mapping.

When to use:
 - When you want to apply the same encoding/scaling on test/inference data.

3. apply_mapping(series, mapping)

What it does: Applies a dict-like mapping to a Pandas series.

When to use:
 - When you're mapping categories to something else (e.g., "M" → 0, "F" → 1").

---------- B. Preprocessor -----------

--- B1. Row & Value Handling ---

1. rows_sampling(df, n)

What it does: Returns a random sample of n rows.

When to use:
 - Quick inspection
 - Reducing huge datasets temporarily.

2. value_rows_remover(df, value, columns)

What it does: Converts a specific value in certain columns to NaN and drops those rows.

When to use:
 - When a column uses a “dummy number” like -1 or 999 meaning “missing”.

3. imputator(df, columns, mode)

Modes: "mean", "median", "mode", "interpolate"

What it does: Fills NaN (and optionally zero) values.

When to use:
 - When you have missing numeric values
 - When zeros represent missing data.

4. distribution_rates(df, columns)

What it does: Shows % of safe / zero / null values.

When to use:
 - When deciding whether to drop or fix a column.
 - When checking data quality.

--- B2. String Cleaning ---

1. batch_string_remover(df, columns, remove)

What it does: Removes specific substrings from text columns.

When to use:
 - Removing units ("mph", "kg", "liters")
 - Cleaning “(USD)” or “kg/m2” types of junk.

2. convert_case(df, columns, mode)

Modes: lower, upper, title, capitalize

What it does: Converts text case.

When to use:
 - When categories are inconsistent ("toyota", "Toyota", "TOYOTA").

3. remove_whitespace(df, columns)

What it does: Removes extra spaces.

When to use:
 - When text has weird spacing issues.

4. remove_irrelevant_characters(df, columns)

What it does: Removes HTML tags and weird symbols.

When to use:
 - When input comes from web scrapers.

--- B3. Dates & Numbers ---

1. standardize_dates(df, columns)

What it does: Turns messy date strings into one standard format (YYYY-MM-DD).

When to use:
 - When date columns use mixed styles like "01-02-23" and "2023 Jan 2".

2. format_numbers(df, columns)

What it does: Removes currency signs, commas → converts to numeric float.

When to use:
 - Cleaning price columns ("$1,299")
 - Cleaning numeric columns that contain text.

--- B4. Category Fixing & Encoding ---

1. normalize_categories(df, columns, mapping, threshold=80)

What it does: Fuzzy-matches category variants and converts them into canonical names.

When to use:
- When you have variants like:
"toyota", "ty0ta", "Toyta", "TOYOTA" → "Toyota"
- When merging datasets with slightly different spellings.

2. FrequencyEncoder(df, columns)

What it does: Replaces categories with how often they appear (0 → rare, 1 → common).

When to use:
 - When categories have many unique values
 - When one-hot would explode dimensionality
 - When category importance roughly correlates with its frequency.

3. OneHotEncoder(df, columns)

What it does: Produces 0/1 dummy columns for each category.

When to use:
 - When categories are few and discrete.
 - Best for tree models or linear models where dummies are useful.

4. TargetEncoder(df, columns, target)

What it does: Replaces each category with the mean target value for that category.

When to use:
 - Handling categorical features in ML tasks
 - Especially when categories are many
 - Especially in high-cardinality columns (job titles, cities).

5. normalize(df, columns)

What it does: MinMax scaling → transforms column into [0,1].

When to use:
 - Before feeding numeric data to ML models that are sensitive to scale
 - Neural networks
 - KNN, SVM
 - Models requiring normalized input.

--- B5. Column Editing ---

1. columns_rename(df, columns, rename_to)

What it does: Rename columns in parallel.

When to use:
 - Standardizing column names
 - Cleaning messy dataset headers.

2. columns_drop(df, columns)

What it does: Drops specified columns.

When to use:
 - Removing irrelevant or useless fields.

3. value_remover(df, value, columns, mode)

Modes: "above", "below", "range"

What it does: Filters rows based on numeric thresholds.

When to use:
 - Removing extreme outliers
 - Keeping rows only above/below certain thresholds
 - Range filtering.

4. dup_row_remover(df, columns, keep)

What it does: Drops duplicate rows based on selected columns.

When to use:
 - When dataset has duplicate entries
 - Removing repeated user IDs, repeated rows.

--- B6. Diagnostics & Helpers ---

1. missing_rows(df, drop_threshold, axis)

What it does: Shows missing percent AND optionally drops rows/columns with too many missing.

When to use:
 - Quick data health check
 - Deciding whether to drop entire columns.

2. unique_items_list(df, columns, count=False)

What it does: Prints unique items in a column.

When to use:
 - Understanding category variety
 - Detecting typos / variants.

3. min_max_finder(df, columns)

What it does: Prints min & max for numeric columns.

When to use:
 - Checking extreme values
 - Validating numeric columns.

--- B7. Saving Data ---

1. save_dataframe(df, output_path, file_format)

Formats: "csv", "json", "jsonl", "excel"

What it does: Saves a cleaned dataframe.

When to use:
 - Final step of your data pipeline
 - Exporting cleaned datasets to use elsewhere.