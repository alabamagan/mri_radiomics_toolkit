import gzip
import base64
import io
import os


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
