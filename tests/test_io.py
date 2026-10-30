"""Unit tests for the io module (NumCompute framework):

This test path verifies the robustness and accuracy of CSV path loading, including delimiters and header skipping,
filling missing values, and raising exception errors (TypeError and ValueError).

Covered test cases:

- Basic CSV loading into Numpy arrays.
- CSV loading with custom delimiters and row skipping.
- Handling and filling missing values.
- Exception raisiong validation: 
    * TypeError for invalid paths
    * ValueError for invalid content

"""

import numpy as np
import pytest 
from numcompute_stream.io import load_csv

def test_load_csv_basic(tmp_path):
    #Create a temporary .csv path
    f = tmp_path / "data.csv"
    f.write_text("1.0,2.0,3.0\n4.0,5.0,6.0")
    
    arr = load_csv(f)
    expected_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    assert np.array_equal(arr, expected_arr)
    
def test_load_csv_custom_delimiter_and_skiprows(tmp_path):
    f = tmp_path / "custom.csv"
    f.write_text("header1\theader2\n10\t20\n30\t40")
    
    arr = load_csv(f, delimiter="\t", skiprows=1, dtype=int)
    expected_arr = np.array([[10, 20], [30, 40]])
    
    assert np.array_equal(arr, expected_arr)
    
def test_load_csv_missing_values(tmp_path):
    f = tmp_path / "missing.csv"
    f.write_text(".,2.0,3.0\n4.0,,6.0")
    
    arr = load_csv(f, missing_values=["", "."], filling_values=np.nan)
    expected_arr = np.array([
        [np.nan, 2.0, 3.0],
        [4.0, np.nan, 6.0]])
    
    assert np.array_equal(arr, expected_arr, equal_nan=True)
        
def test_load_csv_typeerror(tmp_path):
    #An unvalid input path raises TypeError
    with pytest.raises(TypeError):
        load_csv(00)
    
def test_load_csv_valueerror(tmp_path):
    f = tmp_path / "error.csv"
    f.write_text("text,data\nnofloat,here")
    
    #Trying to load text in float raises ValueError
    with pytest.raises(ValueError):
        load_csv(f, dtype=float)