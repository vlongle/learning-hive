'''
File: /record.py
Project: utils
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import pandas as pd


class Record:
    def __init__(self, path: str):
        """
        A convenient class to record data to a csv file.
        We can write a row to the csv file by providing a dictionary
        where key is the column name and value is the value of the column.
        """
        self.path = path
        self.df = pd.DataFrame()

    def write(self, row: dict):
        """
        Write a row to the csv file.
        Key is the column name and value is the value of the column.
        """
        self.df = pd.concat([self.df, pd.DataFrame(row, index=[0])])

    def save(self):
        self.df.to_csv(self.path, index=False)
