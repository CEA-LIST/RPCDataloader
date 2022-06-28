# (C) Copyright 2022 CEA LIST. All Rights Reserved.
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

import torch


def unpickle_tensor(buffer, dtype, shape):
    return torch.frombuffer(buffer, dtype=dtype).view(*shape)


def pickle_tensor(t):
    return unpickle_tensor, (t.contiguous().numpy().view('b'), t.dtype, t.shape)
