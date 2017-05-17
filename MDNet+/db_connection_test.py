#!/usr/bin/env python
# coding=utf-8
__author__ = 'raoqi'

import _init_paths
import os
from lib.vdbc.dataset_factory import VDBC

dtype = 'VOT'
dbpath = os.path.join('data','VOT')
gtpath = dbpath

vdbc = VDBC(dbtype=dtype , dbpath=dbpath, gtpath=gtpath, flush=True)
fc = vdbc.get_frame_count()
print fc
