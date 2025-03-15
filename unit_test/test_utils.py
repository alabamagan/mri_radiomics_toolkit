import string
import unittest
import random
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from mri_radiomics_toolkit.utils import compress, decompress, is_compressed, unify_dataframe_levels, StateManager
import warnings


class Test_StringCompression(unittest.TestCase):
    def test_compress_decompress(self):
        """Compression and decompression should be inverses"""
        text = 'This is a unit_test string'
        compressed = compress(text)
        decompressed = decompress(compressed)
        self.assertEqual(text, decompressed)

    def test_long_string_compressiong(self):
        """Compress and decompress a long random string"""
        # Generate a random string of at least 1000 characters
        length = random.randint(1000, 10000)
        text = ''.join(random.choices(string.ascii_letters + string.digits, k=length))

        compressed = compress(text)
        decompressed = decompress(compressed)
        self.assertEqual(text, decompressed)

    def test_invalid_input(self):
        """Decompression should return None for invalid input"""
        self.assertRaises(ArithmeticError,
                          decompress, 'invalid input')

    def test_empty_string(self):
        """Compress and decompress empty string"""
        text = ''
        compressed = compress(text)
        decompressed = decompress(compressed)
        self.assertEqual(text, decompressed)

    def test_compressed_string(self):
        text = 'This is a unit_test string'
        compressed = compress(text)
        self.assertTrue(is_compressed(compressed))
        self.assertFalse(is_compressed(text))


class TestUnifyDataFrameLevels(unittest.TestCase):
    """Test function for unifying DataFrame levels"""

    def setUp(self):
        """Set up test data"""
        # Single-level header
        self.df1 = pd.DataFrame({
            'RHO193_20201019': [1.0, 2.0, 3.0],
            '1411_20160425': [4.0, 5.0, 6.0]
        })

        # Two-level header
        self.df2 = pd.DataFrame({
            ('1347_20150803', 'C1'): [7.0, 8.0, 9.0],
            ('1347_20150803', 'C2'): [10.0, 11.0, 12.0],
            ('RHO156_20200803', 'C1'): [13.0, 14.0, 15.0]
        })

        # Three-level header
        self.df3 = pd.DataFrame({
            ('1280_20151110', 'C1', 'extra'): [16.0, 17.0, 18.0],
            ('1280_20151110', 'C2', 'extra'): [19.0, 20.0, 21.0]
        })

        # Set feature names as index
        feature_names = [
            'original_firstorder_Mean',
            'original_firstorder_Median',
            'original_shape_Volume'
        ]
        self.df1.index = feature_names
        self.df2.index = feature_names
        self.df3.index = feature_names

    def test_single_level_dataframe(self):
        """Test single-level DataFrame"""
        result = unify_dataframe_levels(self.df1)
        self.assertFalse(isinstance(result.columns, pd.MultiIndex))
        self.assertEqual(len(result.columns), 2)  # Verify number of columns
        self.assertListEqual(result.columns.tolist(), self.df1.columns.tolist())

    def test_two_level_dataframe(self):
        """Test two-level DataFrame"""
        result = unify_dataframe_levels(self.df2)
        self.assertTrue(isinstance(result.columns, pd.MultiIndex))
        self.assertEqual(result.columns.nlevels, 2)  # Verify number of levels in columns

    def test_three_level_dataframe(self):
        """Test three-level DataFrame"""
        result = unify_dataframe_levels(self.df3)
        self.assertTrue(isinstance(result.columns, pd.MultiIndex))
        self.assertEqual(result.columns.nlevels, 3)  # Verify number of levels in columns

    def test_mixed_level_dataframe(self):
        """Test concatenation and unification of mixed-level DataFrames"""
        dfs = [self.df1, self.df2, self.df3]
        result = pd.concat(dfs, axis=1)  # Concatenate along columns
        result = unify_dataframe_levels(result)

        # Verify unified levels
        self.assertTrue(isinstance(result.columns, pd.MultiIndex))
        self.assertEqual(result.columns.nlevels, 3)

        # Check if single-level columns were correctly padded
        self.assertEqual(result.loc['original_firstorder_Mean', ('RHO193_20201019', 'Unknown', 'Unknown')], 1.0)

        # Check if two-level columns were correctly padded
        self.assertEqual(result.loc['original_firstorder_Mean', ('1347_20150803', 'C1', 'Unknown')], 7.0)

        # Check if three-level columns remain unchanged
        self.assertEqual(result.loc['original_firstorder_Mean', ('1280_20151110', 'C1', 'extra')], 16.0)

    def test_empty_dataframe(self):
        """Test empty DataFrame"""
        empty_df = pd.DataFrame()
        result = unify_dataframe_levels(empty_df)
        self.assertTrue(result.empty)

    def test_index_unification(self):
        """Test index unification (axis=0)"""
        # Create DataFrames with different level indices
        df1 = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        df2 = pd.DataFrame({'B': [4, 5, 6]}, index=[('x', '1'), ('y', '2'), ('z', '3')])

        # Concatenate and unify index
        result = pd.concat([df1, df2], axis=0)
        result = unify_dataframe_levels(result, axis=0)

        # Verify unified index levels
        self.assertTrue(isinstance(result.index, pd.MultiIndex))
        self.assertEqual(result.index.nlevels, 2)

    def test_preserve_dtypes(self):
        """Test data type preservation"""
        # Create DataFrame with different data types
        df = pd.DataFrame({
            'float_col': [1.0, 2.0, 3.0],
            ('int_col', 'a'): [1, 2, 3],
            ('str_col', 'b'): ['x', 'y', 'z']
        })

        result = unify_dataframe_levels(df)

        # Check data types
        self.assertEqual(result.dtypes[('float_col', 'Unknown')].name, 'float64')
        self.assertEqual(result.dtypes[('int_col', 'a')].name, 'int64')
        self.assertEqual(result.dtypes[('str_col', 'b')].name, 'object')

    def test_custom_level_names(self):
        """Test custom level names"""
        result = unify_dataframe_levels(self.df2, level_names=["Level_1", "Level_2"])
        self.assertEqual(result.columns.names, ["Level_1", "Level_2"])

    def test_radiomics_feature_structure(self):
        """Test complete structure of radiomics features"""
        dfs = [self.df1, self.df2, self.df3]
        result = pd.concat(dfs, axis=1)
        result = unify_dataframe_levels(result)

        # Convert feature names to multi-level index
        new_index = [o.split('_') for o in result.index]
        new_index = pd.MultiIndex.from_tuples(new_index,
                                              names=('Pre-processing', 'Feature_Group', 'Feature_Name'))
        result.index = new_index

        # Check index structure
        self.assertEqual(result.index.names, ('Pre-processing', 'Feature_Group', 'Feature_Name'))
        self.assertEqual(result.index.nlevels, 3)

        # Check for specific features' existence
        self.assertTrue(('original', 'firstorder', 'Mean') in result.index)
        self.assertTrue(('original', 'shape', 'Volume') in result.index)

    def test_transposed_dataframe(self):
        """Test transposed DataFrame"""
        # Transpose the original DataFrame
        transposed_df = self.df2.T  # Transpose to make the headers the index
        result = unify_dataframe_levels(transposed_df, axis=0)

        # Verify unified index levels (since we transposed, the index is now the focus)
        self.assertTrue(isinstance(result.index, pd.MultiIndex))
        self.assertEqual(result.index.nlevels, 2)

        # Verify data integrity
        self.assertEqual(result.loc[('1347_20150803', 'C1'), 'original_firstorder_Mean'], 7.0)
        self.assertEqual(result.loc[('RHO156_20200803', 'C1'), 'original_shape_Volume'], 15.0)


class TestClass:
    """A test class for checking complex object serialization"""
    def __init__(self, value):
        self.value = value
        
    def __eq__(self, other):
        if not isinstance(other, TestClass):
            return False
        return self.value == other.value


class TestStateManager(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for testing

        TODO: Add support for tuple
        """
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "state.tar.gz"
        
        # Create a test state with various data types
        self.test_state = {
            'string': 'test',
            'number': 42,
            'array': np.array([1, 2, 3]),
            'list': [1, 2, 3],
            'dict': {'a': 1, 'b': 2},
            'tuple': (1, 2, 3), # Tuple is not supported
            'none': None,
            'bool': True
        }
    
    def tearDown(self):
        """Clean up temporary files after tests"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_save_load_basic_types(self):
        """Test saving and loading basic Python types"""
        # Save state
        StateManager.save_state(self.test_state, self.test_file)
        
        # Verify file exists
        self.assertTrue(self.test_file.exists())
        
        # Load state
        loaded_state = StateManager.load_state(self.test_file)
        
        # Check basic types
        self.assertEqual(loaded_state['string'], self.test_state['string'])
        self.assertEqual(loaded_state['number'], self.test_state['number'])
        self.assertEqual(loaded_state['list'], self.test_state['list'])
        self.assertEqual(loaded_state['dict'], self.test_state['dict'])
        self.assertEqual(loaded_state['bool'], self.test_state['bool'])
    
    def test_save_load_numpy_array(self):
        """Test saving and loading of numpy arrays"""
        # Create a state with numpy arrays
        array_state = {
            'array1d': np.array([1, 2, 3]),
            'array2d': np.array([[1, 2], [3, 4]]),
            'array_float': np.array([1.1, 2.2, 3.3])
        }
        
        # Save state
        StateManager.save_state(array_state, self.test_file)
        
        # Load state
        loaded_state = StateManager.load_state(self.test_file)
        
        # Check arrays
        np.testing.assert_array_equal(loaded_state['array1d'], np.array([1, 2, 3]))
        np.testing.assert_array_equal(loaded_state['array2d'], np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_almost_equal(loaded_state['array_float'], np.array([1.1, 2.2, 3.3]))
    
    def test_save_load_pandas_df(self):
        """Test saving and loading of pandas DataFrames"""
        # Create a state with pandas DataFrames
        df_state = {
            'df': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
            'series': pd.Series([5, 6, 7], name='test_series')
        }
        
        # Save state
        StateManager.save_state(df_state, self.test_file)
        
        # Load state
        loaded_state = StateManager.load_state(self.test_file)
        
        # Check DataFrames
        pd.testing.assert_frame_equal(loaded_state['df'], df_state['df'])
        pd.testing.assert_series_equal(loaded_state['series'], df_state['series'])
    
    def test_save_load_nested_structures(self):
        """Test saving and loading of nested data structures"""
        # Create a state with nested structures
        nested_state = {
            'level1': {
                'level2': {
                    'string': 'nested',
                    'number': 99,
                    'array': np.array([7, 8, 9])
                }
            }
        }
        
        # Save state
        StateManager.save_state(nested_state, self.test_file)
        
        # Load state
        loaded_state = StateManager.load_state(self.test_file)
        
        # Check nested structure
        self.assertEqual(loaded_state['level1']['level2']['string'], 'nested')
        self.assertEqual(loaded_state['level1']['level2']['number'], 99)
        np.testing.assert_array_equal(loaded_state['level1']['level2']['array'], np.array([7, 8, 9]))
    
    def test_save_load_empty_state(self):
        """Test saving and loading of an empty state"""
        # Save empty state
        StateManager.save_state({}, self.test_file)
        
        # Load state
        loaded_state = StateManager.load_state(self.test_file)
        
        # Check empty state
        self.assertEqual(loaded_state, {})
    
    def test_save_load_none_values(self):
        """Test saving and loading of states with None values"""
        # Create a state with None values
        none_state = {
            'none1': None,
            'none2': None,
            'valid': 'data'
        }
        
        # Save state
        StateManager.save_state(none_state, self.test_file)
        
        # Load state
        loaded_state = StateManager.load_state(self.test_file)
        
        # Check None values
        self.assertIsNone(loaded_state['none1'])
        self.assertIsNone(loaded_state['none2'])
        self.assertEqual(loaded_state['valid'], 'data')
    
    def test_save_load_complex_objects(self):
        """Test saving and loading of complex Python objects"""
        # Create a state with complex objects
        complex_state = {
            'object': TestClass(42),
            'nested_object': {'obj': TestClass(99)}
        }
        
        # Save state
        StateManager.save_state(complex_state, self.test_file)
        
        # Load state
        loaded_state = StateManager.load_state(self.test_file)
        
        # Check complex objects
        self.assertEqual(loaded_state['object'].value, complex_state['object'].value)
        self.assertEqual(loaded_state['nested_object']['obj'].value, complex_state['nested_object']['obj'].value)
    
    def test_save_load_large_arrays(self):
        """Test saving and loading of large numpy arrays"""
        # Create a state with large arrays
        large_array = np.random.rand(1000, 1000)
        large_state = {
            'large_array': large_array
        }
        
        # Save state
        StateManager.save_state(large_state, self.test_file)
        
        # Load state
        loaded_state = StateManager.load_state(self.test_file)
        
        # Check large array
        np.testing.assert_array_almost_equal(loaded_state['large_array'], large_array)
    
    def test_save_load_mixed_types(self):
        """Test saving and loading state with mixed data types"""
        mixed_state = {
            'string': 'test',
            'number': 42,
            'array': np.array([1, 2, 3]),
            'df': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
            'list': [1, 2, 3],
            'dict': {'a': 1, 'b': 2},
            'none': None,
            'bool': True,
            'tuple': (1, 2) # Tuple 
        }
        
        # Save state
        StateManager.save_state(mixed_state, self.test_file)
        
        # Load state
        loaded_state = StateManager.load_state(self.test_file)
        
        # Check all types
        self.assertEqual(loaded_state['string'], 'test')
        self.assertEqual(loaded_state['number'], 42)
        np.testing.assert_array_equal(loaded_state['array'], np.array([1, 2, 3]))
        pd.testing.assert_frame_equal(loaded_state['df'], pd.DataFrame({'A': [1, 2], 'B': [3, 4]}))
        self.assertEqual(loaded_state['list'], [1, 2, 3])
        self.assertEqual(loaded_state['dict'], {'a': 1, 'b': 2})
        self.assertIsNone(loaded_state['none'])
        self.assertTrue(loaded_state['bool'])
        # self.assertEqual(loaded_state['tuple'], (1, 2))
        
    def test_save_load_with_logger(self):
        """Test saving and loading state with MNTSLogger objects"""
        from mnts.mnts_logger import MNTSLogger
        
        # Create a state with a logger
        state_with_logger = {
            'data': 'test data',
            '_logger': MNTSLogger['test_logger'],
            'nested': {
                'value': 42,
                'inner': np.array([1, 3, 4])
            }
        }
        
        # Save state
        StateManager.save_state(state_with_logger, self.test_file)
        
        # Load state
        loaded_state = StateManager.load_state(self.test_file)
        
        # Check that data was saved but loggers were skipped
        self.assertEqual(loaded_state['data'], 'test data')
        self.assertNotIn('_logger', loaded_state)
        self.assertEqual(loaded_state['nested']['value'], 42)
        self.assertTupleEqual(tuple(loaded_state['nested']['inner']),
                              tuple(state_with_logger['nested']['inner']))
        self.assertNotIn('_logger', loaded_state['nested'])
        
    def test_save_load_nested_dictionaries(self):
        """Test saving and loading nested dictionaries"""
        nested_state = {
            'level1': {
                'level2': {
                    'level3': {
                        'data': 'deep data',
                        'array': np.array([1, 2, 3])
                    }
                },
                'sibling': {
                    'data': 'sibling data'
                }
            }
        }
        
        # Save state
        StateManager.save_state(nested_state, self.test_file)
        
        # Load state
        loaded_state = StateManager.load_state(self.test_file)
        
        # Check nested structure
        self.assertEqual(loaded_state['level1']['level2']['level3']['data'], 'deep data')
        np.testing.assert_array_equal(loaded_state['level1']['level2']['level3']['array'], np.array([1, 2, 3]))
        self.assertEqual(loaded_state['level1']['sibling']['data'], 'sibling data')
        
    def test_save_to_directory_load_from_file(self):
        """Test saving to a directory and loading from the resulting file"""
        # Save state to directory
        StateManager.save_state(self.test_state, self.test_dir)
        
        # Verify file exists
        expected_file = self.test_dir / "state.tar.gz"
        self.assertTrue(expected_file.exists())
        
        # Load state from file
        loaded_state = StateManager.load_state(expected_file)
        
        # Check basic types
        self.assertEqual(loaded_state['string'], 'test')
        self.assertEqual(loaded_state['number'], 42)
        
    def test_load_from_directory(self):
        """Test loading from a directory containing state.tar.gz"""
        # Save state to file
        StateManager.save_state(self.test_state, self.test_file)
        
        # Load state from directory
        loaded_state = StateManager.load_state(self.test_dir)
        
        # Check basic types
        self.assertEqual(loaded_state['string'], 'test')
        self.assertEqual(loaded_state['number'], 42)
        
    def test_missing_file(self):
        """Test loading from a non-existent file"""
        non_existent_file = self.test_dir / "non_existent.tar.gz"
        
        # Attempt to load from non-existent file
        with self.assertRaises(FileNotFoundError):
            StateManager.load_state(non_existent_file)