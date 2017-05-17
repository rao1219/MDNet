__author__ = 'raoqi'

import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2

import os
from lib.utils.timer import Timer

class SolverWrapper(object):
    """
    A simple wrapper around Caffe's solver.
    This solver gives us control over the snapshotting process.
    """
    def __init__(self, solver_prototxt, vdbc, output_dir, pretrained_model = None):
        """
        Initialize the solver and the weights of the net with pre-trained model.
        """
        self._vdbc = vdbc
        self._solver = caffe.SGDSolver(solver_prototxt)
        self._output_dir = output_dir

        if pretrained_model is not None:
            self._solver.net.copy_from(pretrained_model)

        self._solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, "rt") as f:
            pb2.text_format.Merge(f.read(), self._solver_param)

        self._solver.net.layers[0].get_VDBC(vdbc)
    
    def get_net(self):
        return self._solver.net

    def snapshot(self):
        """
        Snapshot the training model's weights.
        :return snapshot model name
        """
        net = self._solver.net

        fname = self._solver_param.snapshot_prefix + \
                '_iter_{:d}'.format(self._solver.iter) + \
                '.caffemodel'
        fname = os.path.join(self._output_dir, fname)

        net.save(str(fname))
        print 'Wrote snapshot to: {:s}'.format(fname)

        return fname

    def train_model(self, max_iters, snapshot_iters):
        """
        Train the model with max_iters.
        :return saved model paths
        """
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []

        print "Begin training the model."
        while self._solver.iter < max_iters:
            timer.tic()
            self._solver.step(1)
            timer.toc()

            # print the speed
            if self._solver.iter % 1000 == 0:
                print 'speed: {:.3f}s / iter.'.format(timer.average_time)
            # snapshot the weights
            if self._solver.iter % snapshot_iters == 0:
                last_snapshot_iter = self._solver.iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self._solver.iter:
            model_paths.append(self.snapshot())

        return model_paths

    def finetune(self, iter):
        """Fine-tune the model.
        """
        timer = Timer()

        timer.tic()
        self._solver.step(iter)
        timer.toc()

        print 'speed: {:.3f}s / iter.'.format(timer.total_time / iter)
