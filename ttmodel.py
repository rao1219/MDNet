#!/usr/bin/env python
# coding=utf-8
__author__ = 'raoqi'

import _init_paths
import caffe
import os.path as osp

caffe.set_device(1)
caffe.set_mode_gpu()

solver_text = osp.join('model','solver.prototxt')
model = osp.join('model','vggm.caffemodel')
solver = caffe.SGDSolver(solver_text)
solver.net.copy_from(model)
