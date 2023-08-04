import gzip
import base64
import io
import os
import time

import pandas as pd
from typing import Optional, Union, Any
from multiprocessing import Queue, Manager, Process
from pathlib import Path
from mnts.mnts_logger import MNTSLogger

def compress(in_str):
    r"""Compresses a string using gzip and base64 encodes it.

    Args:
        string (str): The string to compress.

    Returns:
        str: The base64 encoded gzip compression of the input string.
    """
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

    """
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
    """
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
    r"""
    A singleton class that wraps a separate process which writes pandas Series to an Excel file.

    Attributes:
        output_file (str):
            The path to the output Excel file.
        queue (multiprocessing.Queue):
            The queue for communication between processes.
        process (multiprocessing.Process):
            The subprocess that does the writing.
    """
    _instance = None
    def __init__(self, output_file: Union[Path, str]):
        r"""Initialize the ExcelWriterProcess with an output file path.

        Args:
            output_file (str): The path to the output Excel file.
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
        r"""Send a pandas Series to the writer subprocess to write it to the Excel file.

        Args:
            data (pd.Series or pd.DataFrame): The pandas Series to write.
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
        """The function that runs in the writer subprocess, which writes pandas Series to the Excel file.

        Args:
            queue (multiprocessing.Queue): The queue for receiving pandas Series from the main process.
            output_file (str): The output file where the series will be written to.
        """
        cache = []
        last_flush = time.time()
        LAST_FLUSH_FLAG = False

        # If file already exist, read it to get the index
        mode = 'a' if Path(output_file).is_file() else 'w'
        write_index = mode != 'a'

        # Open the file first
        with pd.ExcelWriter(output_file,
                            engine='openpyxl',
                            mode=mode,
                            if_sheet_exists='overlay' if mode == 'a' else None) as writer:
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
