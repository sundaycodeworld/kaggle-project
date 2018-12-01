# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 08:31:35 2018

@author: zh
"""

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())