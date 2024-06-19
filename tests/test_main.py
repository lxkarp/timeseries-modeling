import unittest
import main


class MainTest(unittest.TestCase):
    def test_add(self):
        # setup
        a = 1
        b = 2
        expected_sum = 3

        # act
        result = main.add(a, b)

        # assert
        self.assertEqual(result, expected_sum)
