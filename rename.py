# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 18:22:49 2022

@author: DGE
"""

import os

path = 'C:/Users/DGE/Desktop/AUO/PPT/'

for i in os.listdir(path):
    name = i.replace('投影片', '')
    os.rename(path + i, path + name)