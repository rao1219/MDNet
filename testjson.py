#!/usr/bin/env python
# coding=utf-8
__author__ = 'raoqi'

import json
import os.path as osp

def _load_json_text():
    with open('./gt_info.json','r') as f:
        data = json.load(f)
        return data


gt = _load_json_text()

for k in gt:
    print k, len(gt[k]), type(gt[k])
