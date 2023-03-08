# (C) Copyright 2022-2023 CEA LIST. All Rights Reserved.
# Contributor(s): Nicolas Granger <nicolas.granger@cea.fr>
#
# This software is governed by the CeCILL-C license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL-C
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-C license and that you accept its terms.

import random

import numpy as np
import torch


def unpickle_tensor(buffer, dtype, shape):
    return torch.frombuffer(buffer, dtype=dtype).view(shape)


def pickle_tensor(t):
    return unpickle_tensor, (t.ravel().numpy().view("b"), t.dtype, t.shape)


pkl_dispatch_table = {torch.Tensor: pickle_tensor}


def set_random_seeds(base_seed, worker_id):
    """Set the seed of default random generators from python, torch and numpy.

    This should be called once on each worker.
    Note that workers may run tasks out of order, so this does not ensure
    reproducibility, only non-redundancy between workers.

    Example:

    >>> base_seed = torch.randint(0, 2**32-1, [1]).item()
    >>> for i, (host, port) in enumerate(workers):
    ...     rpc_async(host, port, set_random_seeds, args=[base_seed, i])
    """

    seed = base_seed + worker_id
    random.seed(seed)
    torch.manual_seed(seed)
    np_seed = torch.utils.data._utils.worker._generate_state(base_seed, worker_id)
    np.random.seed(np_seed)
