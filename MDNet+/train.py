__author__ = 'raoqi'

import _init_paths
import caffe
import os
from lib.vdbc.dataset_factory import VDBC
from tools.solverwrapper import SolverWrapper

ROOT = '.'
dbtype = 'OTB'
dbpath = os.path.join(ROOT, 'data', 'OTB','data')
gtpath = dbpath
output_dir = os.path.join(ROOT, 'model')
solver_prototxt = os.path.join(output_dir, 'solver.prototxt')
pretrained_model = os.path.join(output_dir, 'vggm.caffemodel')

#EXCLUDE_SET = {
#    'vot2014': ['basketball']}


def train_net(pretrained_model, snapshot_iters=1000000):
    vdbc = VDBC(dbtype=dbtype, dbpath=dbpath, gtpath=gtpath, flush=True)
#    vdbc.del_exclude(EXCLUDE_SET['vot2014'])
    print 'VDBC instance built.'

    num_frame = vdbc.get_frame_count()
    max_iters = 64 * 4 * num_frame
    snapshot_iters = 64 * num_frame
    print 'Total number of frames: {}'.format(num_frame)
    print 'Max iterations: {}'.format(max_iters)

    sw = SolverWrapper(solver_prototxt, vdbc, output_dir,pretrained_model)
    print 'Initialization of SolverWrapper finished.'

    sw.train_model(max_iters, snapshot_iters)


def main():
    caffe.set_mode_gpu()
    caffe.set_device(1)

    print 'Model training begins.'
    train_net(pretrained_model)
    print 'Model training finished.'

if __name__ == '__main__':
    main()
