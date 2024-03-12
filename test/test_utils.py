import string
import unittest
import random
from mri_radiomics_toolkit.utils import compress, decompress, is_compressed


class Test_StringCompression(unittest.TestCase):
    def test_compress_decompress(self):
        """Compression and decompression should be inverses"""
        text = 'This is a test string'
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
        text = 'This is a test string'
        compressed = compress(text)
        self.assertTrue(is_compressed(compressed))
        self.assertFalse(is_compressed(text))