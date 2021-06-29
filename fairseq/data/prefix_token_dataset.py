# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class PrefixTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset, prefixes=None):
        super().__init__(dataset)
        self.prefixes = prefixes
        if prefixes is not None:
            self._sizes = np.array(dataset.sizes) + len(self.prefixes)
        else:
            self._sizes = dataset.sizes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.prefixes is not None:
            item = torch.cat([item.new(self.prefixes), item])
        return item

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        n = self.dataset.num_tokens(index)
        if self.token is not None:
            n += len(self.prefixes)
        return n

    def size(self, index):
        n = self.dataset.size(index)
        if self.token is not None:
            n += len(self.prefixes)
        return n
