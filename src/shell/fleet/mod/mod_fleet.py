'''
File: /mod_fleet.py
Project: mod
Created Date: Thursday September 7th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

from shell.fleet.fleet import Fleet, ParallelFleet


class ModFleet(Fleet):
    pass


class ParallelModFleet(ParallelFleet, ModFleet):
    pass
