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
