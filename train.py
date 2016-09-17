__author__ = 'stephen'

import caffe
from lib.vdbc.dataset_factory import VDBC
from tools.solverwrapper import SolverWrapper

ROOT = './'
dbtype = 'OTB'
dbpath = ROOT + 'data/vot2014'
gtpath = ROOT + 'data/vot2014'
output_dir = ROOT + 'model/'
solver_prototxt = ROOT + 'model/solver.prototxt'
pretrained_model = ROOT + 'model/vgg15.caffemodel'

def train_net(pretrained_model, max_iters=60000, snapshot_iters=5000):
    vdbc = VDBC(dbtype=dbtype, dbpath=dbpath, gtpath=gtpath, flush=True)
    print 'VDBC instance built.'
    sw = SolverWrapper(solver_prototxt, vdbc, output_dir, pretrained_model)
    print 'Initialization of SolverWrapper finished.'

    sw.train_model(max_iters, snapshot_iters)

if __name__ == '__main__':
    caffe.set_mode_gpu()

    print 'Model training begins.'
    train_net(pretrained_model)
    print 'Model training finished.'
