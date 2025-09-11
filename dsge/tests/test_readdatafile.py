import os
import unittest
import numpy as np

from dsge.data import read_data_file


class TestReadDataFile(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data/')

    def test_spaces(self):
        data = read_data_file(os.path.join(self.data_dir, 'test_spaces.txt'), ["A", "B", "C", "D"])
        self.assertEqual(data.shape, (2, 4))
    def test_with_header(self):
        data = read_data_file(os.path.join(self.data_dir, 'test_with_header.csv'), ["A", "B", "C"])
        self.assertEqual(data.shape, (2, 3))  # assuming your test file has 2 rows and 3 columns

    def test_no_header(self):
        data = read_data_file(os.path.join(self.data_dir, 'test_no_header.csv'), ["A", "B", "C"])
        self.assertEqual(data.shape, (2, 3))  # adjust to your test file's size

    def test_missing_values(self):
        data = read_data_file(os.path.join(self.data_dir, 'test_missing_values.csv'), ["A", "B", "C"])
        self.assertTrue(data.isnull().values.any())

    def test_dict_input(self):
        data = read_data_file({"file": os.path.join(self.data_dir, 'test_with_header.csv'), "start": '2023-Q1'}, ["A", "B", "C"])
        print(data)
        self.assertEqual(data.shape, (2, 3))  # adjust to your test file's size

    def test_non_existent_file(self):
        data = read_data_file('non_existent.csv', ["A", "B", "C"])
        self.assertTrue(np.isnan(data.values).all())

if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
#import unittest
#import pandas as pd
#import numpy as np
#import warnings
#from datetime import date
#import dsge
#
#read_data_file = dsge.data.read_data_file
#class TestReadDataFile(unittest.TestCase):
#
#    def test_read_data_file(self):
#        # create a dummy data file
#        datafile = "dummy.csv"
#        with open(datafile, "w") as df:
#            df.write("1,2,3\n4,5,6\n")
#
#        # test reading the dummy data file
#        obs_names = ["A", "B", "C"]
#        data = read_data_file(datafile, obs_names)
#        print('\n', data,'*\n')
#        # check that data is a pandas DataFrame object
#        self.assertTrue(isinstance(data, pd.DataFrame))
#
#        # check the dimensions of the DataFrame
#        self.assertEqual(data.shape, (2, 3))
#
#        # check the column names of the DataFrame
#        self.assertEqual(list(data.columns), obs_names)
#
#        # check the values of the DataFrame
#        self.assertTrue(np.allclose(data.values, [[1, 2, 3], [4, 5, 6]]))
#
#        # test the case where the data file doesn't exist
#        datafile = "nonexistent.csv"
#        with self.assertWarns(UserWarning):
#            data = read_data_file(datafile, obs_names)
#
#        # check that data is a pandas DataFrame object
#        self.assertTrue(isinstance(data, pd.DataFrame))
#
#        # check the dimensions of the DataFrame
#        self.assertEqual(data.shape, (100, 3))
#
#        # check the column names of the DataFrame
#        self.assertEqual(list(data.columns), obs_names)
#
#        # check that the DataFrame is filled with NaN values
#        self.assertTrue(np.all(np.isnan(data.values)))
#
#        # test the case where datafile input is a dictionary with optional start parameter
#        datafile = {"file": "dummy.csv", "start": '2022Q1'}
#        obs_names = ["A", "B"]
#        data = read_data_file(datafile, obs_names)
#
#        # check that data is a pandas DataFrame object
#        self.assertTrue(isinstance(data, pd.DataFrame))
#
#        # check the dimensions of the DataFrame
#        self.assertEqual(data.shape, (2, 2))
#
#        # check the column names of the DataFrame
#        self.assertEqual(list(data.columns), obs_names)
#
#        # check the index of the DataFrame
#        expected_index = pd.period_range(start="2022Q1", freq="Q", periods=2)
#        self.assertListEqual(list(data.index), list(expected_index))
#
#if __name__ == '__main__':
#    unittest.main()
