'''
File: /data_fleet.py
Project: data
Created Date: Friday September 8th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

from shell.fleet.fleet import Fleet, ParallelFleet


class DataFleet(Fleet):
    pass


class ParallelDataFleet(ParallelFleet, DataFleet):
    pass
