__author__ = 'stephen'

import caffe
import os
from lib.vdbc.dataset_factory import VDBC
from tools.solverwrapper import SolverWrapper

ROOT = '.'
dbtype = 'OTB'
dbpath = os.path.join(ROOT, 'data', 'OTB')
gtpath = dbpath
output_dir = os.path.join(ROOT, 'model')
solver_prototxt = os.path.join(output_dir, 'solver.prototxt')
pretrained_model = os.path.join(output_dir, 'vgg16.caffemodel')


def train_net(pretrained_model, max_iters=60000, snapshot_iters=5000):
    vdbc = VDBC(dbtype=dbtype, dbpath=dbpath, gtpath=gtpath, flush=True)
    print 'VDBC instance built.'
    sw = SolverWrapper(solver_prototxt, vdbc, output_dir, pretrained_model)
    print 'Initialization of SolverWrapper finished.'

    sw.train_model(max_iters, snapshot_iters)


def main():
    caffe.set_mode_gpu()

    print 'Model training begins.'
    train_net(pretrained_model)
    print 'Model training finished.'

if __name__ == '__main__':
    main()
