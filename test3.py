'''
File: /test3.py
Project: learning-hive
Created Date: Wednesday April 5th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


def add(a=2, b=3):
    return a + b


cfg = {"a": 10}
print(add(**cfg))
