__author__ = 'stephen'

import caffe
import yaml
import numpy as np
from lib.vdbc.dataset_factory import VDBC

class DataLayer(caffe.Layer):
    """MDNet video data layer for training."""

    def get_VDBC(self, vdbc):
        """Get VDBC instance."""
        assert isinstance(vdbc, VDBC), "It is not a VDBC instance."
        self._vdbc = vdbc

    def _shuffle_db_ids(self):
        pass

    def _build_new_minidb(self):
        pass

    def setup(self, bottom, top):
        """Setup the DataLayer."""

        # parse the layer parameter string
        layer_params = yaml.load(self.param_str_)

        self._batch_size = layer_params['batch_size']
        print '[DataLayer] Batch size: {-d}.'.format(self._batch_size)

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(self._batch_size, 3, 107, 107)
        self._name_to_top_map['data'] = idx

        idx += 1
        top[idx].reshape(1)
        self._name_to_top_map['label'] = idx

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        pass

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
