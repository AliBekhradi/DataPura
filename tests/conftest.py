import pytest
import pandas as pd
from main import Preprocessor

"""intialization"""
@pytest.fixture
def prep():
    return Preprocessor()

"""Insights"""
@pytest.fixture
def unique_items_df():
    return pd.DataFrame({
        'city': ['Tehran', 'tehran', 'Isfahan', 'Mashhad', 'Tehran', '', None],
        'gender': ['Male', 'Female', 'male', 'FEMALE', '', None, 'Female'],
        'joined': ['2020-01-01', '', None, '2021-05-20', '', '2022-12-12', None]
    })

@pytest.fixture
def minmax_df():
    return pd.DataFrame({
        'price': [100, 150, 200, 250, 300],
        'discount': [5, 10, 15, 5, 0],
        'quantity': [1, 3, 2, 5, 4]
    })

@pytest.fixture
def missingrows_df():
    return pd.DataFrame({
        "name": ["Ali", "Sara", None, "Reza", "Nima"],
        "age": [25, None, 30, None, 28],
        "email": ["a@example.com", None, None, None, "e@example.com"],
        "score": [90, None, None, None, None]
    })

"""Basics"""
@pytest.fixture
def rows_sampling_df():
    return pd.DataFrame({
        'id': range(100),
        'value': [x * 10 for x in range(100)]
    })

@pytest.fixture
def column_rename_df():
    return pd.DataFrame({
        "first_name": ["Ali", "Sara", "John"],
        "last_name": ["Bekhradi", "Smith", "Doe"],
        "age": [24, 28, 31]
    })

@pytest.fixture
def column_drop_df():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["Tehran", "Dubai", "Berlin"],
        "income": [50000, 60000, 70000]
    })
    
@pytest.fixture
def convert_case_df():
    return pd.DataFrame({
        'name': ['john doe', 'JANE DOE', 'alice Mcdonald'],
        'country': ['usa', 'Germany', 'UAE']
    })
    
@pytest.fixture
def remove_whitespace_df():
    return pd.DataFrame({
        "name": ["Alex  Bolton", " Sara Stacy", "Peter Parker ", " Al  Pacino ", " Sara", "Ali "],
        "city": ["  Tehran  ", "  Los Angeles", "  Dubai ", "Venice  ", " Istanbul", "London "],
        "country": ["Iran ", " USA", " UAE ", "Italy ", "  Turkey", "  UK"]
    })
    
@pytest.fixture
def standardize_date_df():
    return pd.DataFrame({
        "start_date": [
            "1/1/2023",         # MM/DD/YYYY
            "Jan 2, 23",        # Month D, YY
            "2023-03-01",       # Already ISO
            None,               # Null
            "13/07/2023",       # DD/MM/YYYY
            "1st January 2023", # Natural language
            "2025-08-05",       # Correct ISO
            "invalid-date"      # Unparsable
        ]
    })

@pytest.fixture
def format_numbers_df():
    return pd.DataFrame({
        'price': ['$1,200.50', '€3,450.99', '£-720.3', 'invalid', None],
        'revenue': ['1,000', '2,500.5', 'invalid_data', '3000', '$4000'],
        'not_numeric': ['hello', 'world', '!', '123', '456']
    })
    
@pytest.fixture
def remove_irrelavant_characters_df():
    return pd.DataFrame({
        "title": ["Hello <b>World!</b>", "   Extra   spaces   here   ", "Special $$$$ characters"],
        "notes": ["<p>Some text</p>", None, "Clean already"]
    })

@pytest.fixture
def rows_duplicate_remover_df():
    return pd.DataFrame({
        "name": ["Ali", "Ali", "Sara", "Reza", "Sara", "Ali"],
        "city": ["Tehran", "Tehran", "Shiraz", "Tabriz", "Shiraz", "Tehran"],
        "age": [25, 25, 30, 22, 30, 25]
    })
    
@pytest.fixture
def imputation_df():
    return pd.DataFrame({
        "age": [25, 30, None, 40, None],
        "income": [50000, None, 60000, None, 70000],
        "score": [None, None, 80, 85, 90]
    })

@pytest.fixture
def normalization_df():
    return pd.DataFrame({
    "salary": [50000, 60000, 55000, 65000, 52000],
    "bonus": [5000, 7000, 6000, 8000, 5500],
    "experience_years": [2, 5, 3, 7, 4],
    "department": ["IT", "HR", "IT", "Finance", "HR"]
})

@pytest.fixture
def save_dataframe_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [85.5, 92.0, 78.3],
        "passed": [True, True, False]
    })

@pytest.fixture
def freq_encoding_df():
    return pd.DataFrame({
        "department": ["HR", "IT", "Finance", "IT", "HR", "HR", "Finance"],
        "region": ["East", "West", "East", "North", "West", "West", "North"],
        "score": [80, 90, 85, 88, 70, 95, 75]
    })
    
@pytest.fixture
def target_encoding_df():
    return pd.DataFrame({
        "department": ["HR", "IT", "Finance", "IT", "HR", "Finance", "HR", "IT", "Finance", "HR"],
        "region": ["East", "West", "East", "North", "West", "East", "East", "North", "North", "West"],
        "performance_score": [88, 92, 85, 91, 79, 87, 90, 89, 84, 80]
    })
    
@pytest.fixture
def onehot_encoding_df():
    return pd.DataFrame({
        "color": ["Red", "Blue", "Green", "Blue", "Red", "Green", "Red"],
        "size": ["S", "M", "L", "M", "S", "L", "XL"],
        "price": [10.5, 12.0, 9.8, 13.5, 11.0, 10.2, 14.0]
    })