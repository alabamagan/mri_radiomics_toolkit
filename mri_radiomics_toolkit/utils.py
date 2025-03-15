import gzip
import base64
import io
import os
import time
import numpy as np
import pandas as pd
from typing import Optional, Union, Any, Iterable, List
from multiprocessing import Queue, Manager, Process
from pathlib import Path
import json
import pickle
import warnings

import sklearn.linear_model
from mnts.mnts_logger import MNTSLogger
from functools import partial
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(y: Union[np.ndarray, Iterable]) -> np.ndarray:
    r"""Compute the class weights
    
    Class weights are computed based on the reciprocal of the number of samples per class.
    
    .. math::
        W(c) = \lambda \frac{1}{N_c}
        
    where :math:`\lambda` is the global regularizer such that the weight of the class with the
    most member is normalized to 1. 
           
    Returns:
        (np.ndarray) - The class weights with shape equals to y.shape
    """ # noqa
    # Get the unique classes
    classes = np.unique(y)
    # Calculate the class weights
    weights = compute_class_weight('balanced', classes=classes, y=y)
    # normalize weights
    weights = weights / weights.min()
    # create the class weight dictionary
    class_weight_dict = dict(zip(classes, weights))
    # construct the weight vector
    weight_vector = np.array([class_weight_dict[i] for i in y])
    return weight_vector


def compress(in_str):
    r"""Compresses a string using gzip and base64 encodes it. This function is used to compress
    the settings for controller intiializastion. However, this is no longer used. Settings are
    saved as file-stream binaries now.

    Args:
        string (str): The string to compress.

    Returns:
        str: The base64 encoded gzip compression of the input string.
    
    .. note::
        This function is no longer used. See :class:`Controller` for more.
        
    """ # noqa
    compressed_stream = io.BytesIO()
    gzip_stream = gzip.GzipFile(fileobj=compressed_stream, mode='wb')
    gzip_stream.write(in_str.encode('utf-8'))
    gzip_stream.close()
    compressed = compressed_stream.getvalue()
    b64_content = base64.b64encode(compressed)
    ascii_content = b64_content.decode('ascii')
    return ascii_content


def decompress(string):
    r"""Decompresses a gzipped and base64 encoded string.

    Args:
        string (str):
            The compressed string to decompress.

    Returns:
        str:
            The original uncompressed string. Return None if the input is invalid.
            
    Raise: 
        OSError:
            Raised when input string is not a valid compressed string.
    """ # noqa
    try:
        b64_decoded = base64.b64decode(string)
        decompressed_stream = io.BytesIO(b64_decoded)
        gzip_stream = gzip.GzipFile(fileobj=decompressed_stream, mode='rb')
        return gzip_stream.read().decode('utf-8')
    except OSError:
        raise ArithmeticError('Invalid input, not a compressed string')


def is_compressed(string):
    """Checks if a string is compressed.

    Args:
        string (str): The string to check.
    Returns:
        bool: True if the string is compressed, False otherwise.
    """ # noqa
    # Attempt to decompress the string
    try:
        decompress(string)
        return True
    except:
        return False


def zipdir(path, ziph):
    r"""
    Helper function to zip the norm state directory
    """
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.join(root, file).replace(path, ''))


class ExcelWriterProcess:
    r"""A singleton class for managing a separate process to write pandas Series or DataFrame to an Excel file.

    This class manages an independent process that writes pandas data structures to an Excel file.
    The data is passed via a multiprocessing queue, which is then written to the Excel file by the
    separate process. This class is designed as a singleton, so only one instance will ever exist
    during the program execution and that read-write is thread safe.

    Attributes:
        output_file (Union[Path, str]):
            The file path to the output Excel file. This can be a string or a Path object.
        queue (multiprocessing.Queue):
            The multiprocessing queue used to pass data between the main process and the writing process.
        process (multiprocessing.Process):
            The separate process that performs the writing of data to the Excel file.

    Examples:
    >>>from mri_radiomics_toolkit.utils import ExcelWriterProcess
    >>># initialize the writer process with the output file
    >>>writer = ExcelWriterProcess('output.xlsx')
    >>>
    >>># start the writer process
    >>>writer.start()
    >>>
    >>># create a pandas DataFrame
    >>>df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>>
    >>># send the DataFrame to the writer process
    >>>writer.write(df)
    >>>
    >>># stop the writer process
    >>>writer.stop()
    """ # noqa
    _instance = None
    def __init__(self, output_file: Union[Path, str]):
        r"""Initialize the ExcelWriterProcess instance with an output file path.

        Args:
            output_file (Union[Path, str]):
                The file path to the output Excel file. This can be a string or a Path object.
        """
        self.manager = Manager()
        self.queue = self.manager.Queue()
        self.output_file = output_file
        self.process = Process(target=self._run, args=(self.queue, self.output_file))
        self.__class__.logger = MNTSLogger['ExcelWriterProcess']

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def start(self):
        r"""Start the writer subprocess."""
        self.process.start()

    @classmethod
    def write(cls, data: Union[pd.Series, pd.DataFrame]) -> None:
        r"""Sends a pandas Series or DataFrame to the writer subprocess.

        The data is placed in a queue and is written to the Excel file by the writer subprocess.

        Args:
            data (Union[pd.Series, pd.DataFrame]):
                The pandas Series or DataFrame to write.

        Raises:
            ArithmeticError:
                If this method is called before an instance of ExcelWriterProcess is created.
        """
        if cls._instance is None:
            raise ArithmeticError("Write must only be called after a writer instance is created.")
        # if not cls._instance.process.is_alive():
        #     raise RuntimeError("ExcelWriterProcess.write is called but no process is alive. Have you called start()?")
        instance = cls._instance
        instance.queue.put(data)

    def stop(self) -> None:
        """Stop the writer subprocess and wait for it to finish."""
        self.queue.put(None)  # Signal to stop processing
        self.process.join()  # Wait for the process to finish

    @staticmethod
    def _run(queue: Queue, output_file: Union[str, Path]) -> None:
        r"""Writes pandas Series or DataFrame to the Excel file in the writer subprocess.

        This function runs in the separate writer process and writes data from the queue to the
        Excel file. The data is written in batches to optimize performance.

        Args:
            queue (multiprocessing.Queue):
                The queue for receiving pandas Series or DataFrame from the main process.
            output_file (Union[str, Path]):
                The file path to the output Excel file. This can be a string or a Path object.
        """
        cache = []
        last_flush = time.time()
        LAST_FLUSH_FLAG = False

        # If file already exist, read it to get the index
        output_file = Path(output_file)
        mode = 'a' if output_file.is_file() else 'w'
        write_index = mode != 'a'

        # Determine which writer to use
        if output_file.suffix == '.xlsx':
            writer_class = pd.ExcelWriter
            kwargs = {
                'engine': 'openpyxl',
                'mode': mode,
                'if_sheet_exists': 'overlay' if mode == 'a' else None
            }
        elif output.suffix == '.hdf':
            raise NotImplementedError


        # Open the file first
        with writer_class(output_file, **kwargs) as writer:
            while True:
                try:
                    logger = MNTSLogger['ExcelWriterProcess']
                    data = queue.get()
                    logger.info(f"Got data: {data}")
                    time_passed = time.time() - last_flush
                    if data is None:  # Check if it's the signal to stop processing
                        # Check if there's still something thats not written in cache
                        if len(cache) > 0:
                            LAST_FLUSH_FLAG = True # This will trigger the program to flush immediately
                        else:
                            break
                    else:
                        cache.append(data)

                    # Because the excel file could be very large, and it would be slow to write the data everytime
                    # a data column arrives. Therefore, we flush the cache every 20 rows.
                    if len(cache) >= 20 or time_passed > 360 or LAST_FLUSH_FLAG:
                        if LAST_FLUSH_FLAG:
                            logger.info("Performing last flush.")
                        df = pd.concat(cache, axis=1)
                        last_flush = time.time()
                        cache.clear()
                        logger.debug(f"Writing data: {df}")

                        # Write data to excel file
                        sheetnames = list(writer.sheets.keys())
                        default_sheet = sheetnames[0] if len(sheetnames) > 1 else 'Sheet1'

                        # Convert pd.Series to pd.DataFrame for writing
                        df.to_excel(
                            writer,
                            index = write_index,  # No need to write index if columns are appended
                            startcol=writer.sheets[default_sheet].max_column if mode == 'a' else 0
                        )
                        logger.debug(f"Done writing data: {df}")

                    if LAST_FLUSH_FLAG:
                        break
                except BrokenPipeError:
                    continue
                except Exception as e:
                    logger = ExcelWriterProcess._instance.logger
                    if not logger is None:
                        logger.error(f"An error occurred: {e}")
                    else:
                        print(f"An error occurred: {e}")


def unify_dataframe_levels(df: pd.DataFrame,
                           axis: int =1,
                           level_names: Optional[Union[List[str], str]] = None,
                           placeholder: str = 'Unknown') -> pd.DataFrame:
    """
    Unify the levels of a DataFrame's columns or index to ensure consistency.

    This function adjusts the levels of a DataFrame's columns or index to have a consistent
    number of levels. If the levels are tuples with varying lengths, the shorter tuples
    are padded with empty strings (`''`) to match the maximum number of levels. Optionally,
    custom level names can be provided.

    Args:
        df (pandas.DataFrame):
            The input DataFrame, which may have mixed levels in its columns or index.
        axis (int, optional):
            The axis to unify. Use `1` for columns (default) or `0` for the index.
        level_names (list of str, optional):
            Names for the levels in the MultiIndex. If not provided, default names
            such as `Level_0`, `Level_1`, etc., will be used.

    Returns:
        pandas.DataFrame:
            A new DataFrame with unified levels in the specified axis (columns or index).

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     [[1, 2], [3, 4]],
        ...     columns=[("A", "X"), "B"]
        ... )
        >>> unify_dataframe_levels(df)
           A        B
           X
           1  2
        >>> unify_dataframe_levels(df, level_names=["First", "Second"]).columns
        MultiIndex([('A', 'X'),
                    ('B', '')],
                   names=['First', 'Second'])
    """
    if axis == 1:  # Columns
        # Get the columns
        items = df.columns.tolist()

        # Determine the maximum number of levels
        max_levels = 1
        for item in items:
            if isinstance(item, tuple):
                max_levels = max(max_levels, len(item))

        # If all items are strings (single level) and no level_names specified, return as is
        if max_levels == 1 and level_names is None:
            return df

        # Create new tuples with consistent levels
        new_items = []
        for item in items:
            if isinstance(item, tuple):
                # If tuple has fewer levels than max_levels, pad with empty strings
                if len(item) < max_levels:
                    new_items.append(item + (placeholder,) * (max_levels - len(item)))
                else:
                    new_items.append(item)
            else:
                # Single level - convert to tuple with empty strings for additional levels
                new_items.append((item,) + (placeholder,) * (max_levels - 1))

        # Set level names if provided
        if level_names is None:
            level_names = [f'Level_{i}' for i in range(max_levels)]

        # Create a new DataFrame with the unified columns
        result_df = df.copy()
        result_df.columns = pd.MultiIndex.from_tuples(new_items, names=level_names)

    else:  # Index (axis=0)
        # Get the index
        items = df.index.tolist()

        # Determine the maximum number of levels
        max_levels = 1
        for item in items:
            if isinstance(item, tuple):
                max_levels = max(max_levels, len(item))

        # If all items are strings (single level) and no level_names specified, return as is
        if max_levels == 1 and level_names is None:
            return df

        # Create new tuples with consistent levels
        new_items = []
        for item in items:
            if isinstance(item, tuple):
                # If tuple has fewer levels than max_levels, pad with empty strings
                if len(item) < max_levels:
                    new_items.append(item + ('',) * (max_levels - len(item)))
                else:
                    new_items.append(item)
            else:
                # Single level - convert to tuple with empty strings for additional levels
                new_items.append((item,) + ('',) * (max_levels - 1))

        # Set level names if provided
        if level_names is None:
            level_names = [f'Level_{i}' for i in range(max_levels)]

        # Create a new DataFrame with the unified index
        result_df = df.copy()
        result_df.index = pd.MultiIndex.from_tuples(new_items, names=level_names)

    return result_df


class StateManager:
    """A utility class for managing state saving and loading in a more robust way.
    
    This class provides methods to save and load states of objects that may contain
    non-serializable components. It handles different types of data appropriately:
    - Basic Python types are saved as JSON
    - Numpy arrays are saved using numpy.save
    - Complex Python objects are saved using pickle (if serializable)
    
    States are saved to a single compressed tarball (.tar.gz) file, which contains:
    - state.json: Basic Python types
    - *.npy: NumPy arrays
    - *.parquet: Pandas DataFrames/Series
    - *.pkl: Pickled complex objects
    - Subdirectories for nested dictionaries
    """
    
    @staticmethod
    def save_state(state_dict: dict, save_path: Path):
        """Save a state dictionary to a compressed tarball.
        
        Args:
            state_dict: Dictionary containing the state to save
            save_path: Path to save the state to. If it's a directory, 
                      a file named 'state.tar.gz' will be created in it.
                      If the path doesn't end with .tar.gz, it will be 
                      appended automatically.
        """
        import tarfile
        import tempfile
        import os
        
        save_path = Path(save_path)
        
        # If save_path is a directory, create a file named 'state.tar.gz' in it
        if save_path.is_dir():
            save_path = save_path / 'state.tar.gz'
        # Ensure the file has .tar.gz suffix
        elif not str(save_path).endswith('.tar.gz'):
            save_path = save_path.with_suffix('.tar.gz')
        
        # Ensure the parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary directory to store files before compression
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Save basic types as JSON
            json_state = {}
            for key, value in state_dict.items():
                # Skip logger objects
                if key == '_logger' or (isinstance(value, dict) and '_logger' in value):
                    continue
                    
                if isinstance(value, (str, int, float, bool, type(None))):
                    json_state[key] = value
                elif isinstance(value, (list, tuple)):
                    # Check if all elements are basic types
                    if all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
                        json_state[key] = list(value)  # Convert tuple to list for JSON
                    else:
                        # Try to pickle complex lists/tuples
                        try:
                            with open(temp_dir_path / f"{key}.pkl", 'wb') as f:
                                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                        except Exception as e:
                            warnings.warn(f"Could not pickle {key}: {str(e)}")
                elif isinstance(value, dict):
                    # Recursively save nested dictionaries
                    nested_dir = temp_dir_path / key
                    nested_dir.mkdir(exist_ok=True)
                    try:
                        StateManager._save_nested_dict(value, nested_dir)
                    except Exception as e:
                        warnings.warn(f"Could not save nested dictionary {key}: {str(e)}")
                elif isinstance(value, np.ndarray):
                    # Save numpy arrays separately
                    np.save(temp_dir_path / f"{key}.npy", value)
                elif isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                    # Save pandas objects to parquet
                    try:
                        if isinstance(value, pd.DataFrame):
                            value.to_parquet(temp_dir_path / f"{key}.parquet")
                        else:
                            value.to_frame().to_parquet(temp_dir_path / f"{key}.parquet")
                    except Exception as e:
                        warnings.warn(f"Could not save {key} to parquet: {str(e)}. Trying pickle instead.")
                        try:
                            with open(temp_dir_path / f"{key}.pkl", 'wb') as f:
                                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                        except Exception as e:
                            warnings.warn(f"Could not pickle {key}: {str(e)}. Object will be skipped.")
                else:
                    # Try to pickle other objects
                    try:
                        with open(temp_dir_path / f"{key}.pkl", 'wb') as f:
                            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception as e:
                        warnings.warn(f"Could not pickle {key}: {str(e)}. Object will be skipped.")

            # Save JSON state
            with open(temp_dir_path / "state.json", 'w') as f:
                json.dump(json_state, f, indent=2)
            
            # Create a tarball with all the files
            with tarfile.open(save_path, "w:gz") as tar:
                # Add all files from the temporary directory to the tarball
                for file_path in temp_dir_path.glob("**/*"):
                    if file_path.is_file():
                        # Get the relative path from temp_dir_path
                        rel_path = file_path.relative_to(temp_dir_path)
                        tar.add(file_path, arcname=str(rel_path))
    
    @staticmethod
    def _save_nested_dict(nested_dict: dict, save_dir: Path):
        """Helper method to save nested dictionaries recursively."""
        # Save basic types as JSON
        json_state = {}
        for key, value in nested_dict.items():
            # Skip logger objects
            if key == '_logger':
                continue
                
            if isinstance(value, (str, int, float, bool, type(None))):
                json_state[key] = value
            elif isinstance(value, (list, tuple)):
                # Check if all elements are basic types
                if all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
                    json_state[key] = list(value)  # Convert tuple to list for JSON
                else:
                    # Try to pickle complex lists/tuples
                    try:
                        with open(save_dir / f"{key}.pkl", 'wb') as f:
                            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception as e:
                        warnings.warn(f"Could not pickle {key}: {str(e)}")
            elif isinstance(value, dict):
                # Recursively save nested dictionaries
                nested_dir = save_dir / key
                nested_dir.mkdir(exist_ok=True)
                StateManager._save_nested_dict(value, nested_dir)
            elif isinstance(value, np.ndarray):
                # Save numpy arrays separately
                np.save(save_dir / f"{key}.npy", value)
            elif isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                # Save pandas objects to parquet
                try:
                    if isinstance(value, pd.DataFrame):
                        value.to_parquet(save_dir / f"{key}.parquet")
                    else:
                        value.to_frame().to_parquet(save_dir / f"{key}.parquet")
                except Exception as e:
                    warnings.warn(f"Could not save {key} to parquet: {str(e)}. Trying pickle instead.")
                    try:
                        with open(save_dir / f"{key}.pkl", 'wb') as f:
                            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception as e:
                        warnings.warn(f"Could not pickle {key}: {str(e)}. Object will be skipped.")
            else:
                # Try to pickle other objects
                try:
                    with open(save_dir / f"{key}.pkl", 'wb') as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    warnings.warn(f"Could not pickle {key}: {str(e)}. Object will be skipped.")
        
        # Save JSON state
        with open(save_dir / "state.json", 'w') as f:
            json.dump(json_state, f, indent=2)
    
    @staticmethod
    def load_state(load_path: Path) -> dict:
        """Load a state dictionary from a compressed tarball or directory.
        
        This method loads a previously saved state from the specified path.
        
        If the path is a tarball (.tar.gz file), it will be extracted to a temporary directory
        and the state will be loaded from there.
        
        If the path is a directory, it will look for a state.tar.gz file in the directory.
        If found, it will load from that file. Otherwise, it will try to load directly from
        the directory structure.
        
        Args:
            load_path: Path to the tarball file or directory
            
        Returns:
            Dictionary containing the loaded state
            
        Raises:
            FileNotFoundError: If the specified file or directory does not exist
        """
        import tarfile
        import tempfile
        
        load_path = Path(load_path)
        
        # Check if the path exists
        if not load_path.exists():
            raise FileNotFoundError(f"Path does not exist: {load_path}")
        
        # If load_path is a directory, look for state.tar.gz in it
        if load_path.is_dir():
            tar_path = load_path / 'state.tar.gz'
            if tar_path.exists():
                load_path = tar_path
            else:
                # If state.tar.gz doesn't exist, try to load directly from the directory
                return StateManager._load_from_directory(load_path)
        
        # At this point, load_path should be a file
        if not load_path.is_file():
            raise FileNotFoundError(f"Expected a file but found: {load_path}")
        
        # Create a temporary directory to extract files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Extract the tarball
            try:
                with tarfile.open(load_path, "r:gz") as tar:
                    tar.extractall(path=temp_dir_path)
                
                # Load the state from the extracted files
                return StateManager._load_from_directory(temp_dir_path)
            except tarfile.ReadError as e:
                # If it's not a valid tarball, try to load directly from the path
                warnings.warn(f"Could not read {load_path} as a tarball: {str(e)}. Trying to load directly.")
                if load_path.is_dir():
                    return StateManager._load_from_directory(load_path)
                else:
                    raise
    
    @staticmethod
    def _load_from_directory(load_dir: Path) -> dict:
        """Helper method to load state from a directory."""
        state = {}
        state_json_path = load_dir / "state.json"
        
        # Load JSON state if it exists
        if state_json_path.exists():
            with open(state_json_path, 'r') as f:
                try:
                    state = json.load(f)
                except json.JSONDecodeError as e:
                    warnings.warn(f"Error decoding state.json: {str(e)}. Continuing with empty state.")
        else:
            warnings.warn(f"state.json not found in {load_dir}. Attempting to reconstruct state from other files.")
            
        # Load numpy arrays
        for npy_file in load_dir.glob("*.npy"):
            key = npy_file.stem
            try:
                state[key] = np.load(npy_file, allow_pickle=True)
            except Exception as e:
                warnings.warn(f"Could not load numpy array {key}: {str(e)}")
            
        # Load pandas objects
        for parquet_file in load_dir.glob("*.parquet"):
            key = parquet_file.stem
            try:
                df = pd.read_parquet(parquet_file)
                # Convert back to Series if it's a single column DataFrame with the same name
                if df.shape[1] == 1 and df.columns[0] == key:
                    state[key] = df[key]
                else:
                    state[key] = df
            except Exception as e:
                warnings.warn(f"Could not load parquet file {key}: {str(e)}")
            
        # Load pickled objects
        for pkl_file in load_dir.glob("*.pkl"):
            key = pkl_file.stem
            try:
                with open(pkl_file, 'rb') as f:
                    state[key] = pickle.load(f)
            except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError) as e:
                warnings.warn(f"Could not unpickle {key}: {str(e)}")
                
        # Load nested dictionaries
        for dir_path in [p for p in load_dir.glob("*") if p.is_dir()]:
            key = dir_path.name
            try:
                state[key] = StateManager._load_from_directory(dir_path)
            except Exception as e:
                warnings.warn(f"Error loading nested state from {dir_path}: {str(e)}")
                
        return state
