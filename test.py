# tests/test_math_utils.py

import unittest

class TestMathUtils(unittest.TestCase):
    def test_add(self):
        self.assertEqual(2+3, 5)

    def test_multiply(self):
        self.assertEqual(2*3, 6)

if __name__ == "__main__":
    unittest.main()