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

import io
import pickle
import select
import socket
import struct
import threading
from typing import Callable, TypeVar

import torch
from tblib import pickling_support
from torch.futures import Future

# absolute import required for unpickling
from rpcdataloader.utils import pickle_tensor


pickling_support.install()


_T = TypeVar("_T")


def _serialize(obj, buffer_callback=None):
    buffer = io.BytesIO()
    pickler = pickle.Pickler(
        buffer, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=buffer_callback
    )
    pickler.dispatch_table = {torch.Tensor: pickle_tensor}
    pickler.dump(obj)
    return buffer.getvalue()


def _sock_read(sock, size, buffer=None):
    """Read size bytes from sock."""
    buffer = bytearray(size) if buffer is None else buffer
    received = 0
    while received < size:
        nread = sock.recv_into(memoryview(buffer)[received:])
        if not nread:
            raise RuntimeError("Unexpected socket shutdown.")
        received += nread

    return buffer


def _rpc_send_command(host, port, fut, func, args, kwargs, pin_memory, timeout):
    try:
        payload = _serialize((func, args, kwargs))

        # connect to server
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # send command
            s.sendall(struct.pack("L", len(payload)))
            s.sendall(payload)

            select.select([s], [], [])

            # receive buffers
            payload = _sock_read(s, struct.calcsize("L"))
            (nbuffers,) = struct.unpack("L", payload)
            buffers = []
            for _ in range(nbuffers):
                payload = _sock_read(s, struct.calcsize("L"))
                (n,) = struct.unpack("L", payload)
                b = torch.empty(n, dtype=torch.uint8, pin_memory=pin_memory).numpy()

                _sock_read(s, n, b)

                buffers.append(b)

            # receive object
            payload = _sock_read(s, struct.calcsize("L"))
            (n,) = struct.unpack("L", payload)
            payload = _sock_read(s, n)

        out, err = pickle.loads(payload, buffers=buffers)

        if err:
            fut.set_exception(err)
        else:
            fut.set_result(out)

    except Exception as e:
        fut.set_exception(e)


def rpc_async(
    host: str,
    port: int,
    func: Callable[..., _T],
    args=(),
    kwargs={},
    pin_memory=False,
    timeout=120.0,
) -> Future[_T]:
    """Execute function on remote worker and return the result as a future.

    :param host: rpc worker host
    :param port: rpc worker port
    :param func: function to execute, must be picklable as well as its output
    :param args: positional arguments, must be picklable
    :param kwargs: keword arguments, must be picklable
    :param pin_memory:
        wether buffers of the return value should be allocated in pinned memory.
    :param timeout: timeout in seconds on network operations

    :return: A future that will contain the function return value.
    :rtype: :class:`torch.futures.Future`
    """
    fut = Future()
    t = threading.Thread(
        target=_rpc_send_command,
        args=(host, port, fut, func, args, kwargs, pin_memory, timeout),
    )
    t.start()

    return fut


def _handle_client(sock, parallel_sem):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CORK, 1)

    payload = _sock_read(sock, struct.calcsize("L"))
    (n,) = struct.unpack("L", payload)
    payload = _sock_read(sock, n)
    cmd, args, kwargs = pickle.loads(payload)

    try:
        with parallel_sem:
            out = cmd(*args, **kwargs)
        err = None
    except Exception as e:
        out = None
        err = e

    try:
        buffers = []
        payload = _serialize((out, err), buffer_callback=buffers.append)
    except Exception as e:
        buffers = []
        payload = _serialize((None, e))

    buffers = [memoryview(b).tobytes() for b in buffers]

    sock.sendall(struct.pack("L", len(buffers)))
    for b in buffers:
        sock.sendall(struct.pack("L", len(b)))
        sock.sendall(b)

    sock.sendall(struct.pack("L", len(payload)))
    sock.sendall(payload)

    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CORK, 0)


def run_worker(host: str, port: int, timeout: float = 120, parallel: int = 1, log_hostname=True):
    """Start listening and processing remote procedure calls.

    :param host: interface to bind to (set to '0.0.0.0' for all interfaces)
    :param port: port to bind to
    :param timeout: timeout on network transfers from/to client
    :param parallel: max number procedures executing concurrently
    :param log_hostname: print `<hostname>`:`<port>` once ready

    .. warning::
       The workers neither implement authentication nor encryption, any
       user on the network can send arbitrary commands or may listen to the
       traffic from/to the worker.

    .. note::
       * each request is processed in a separate thread
       * network transfers may overlap regardless of :attr:`parallel` argument.
    """
    parallel_sem = threading.Semaphore(parallel)
    with socket.create_server(
        (host, port), family=socket.AF_INET, backlog=2048, reuse_port=True
    ) as sock:
        if log_hostname:
            print(f"{socket.gethostname()}:{port}")

        while True:
            client_sock, _ = sock.accept()
            client_sock.settimeout(timeout)
            t = threading.Thread(
                target=_handle_client, args=[client_sock, parallel_sem]
            )
            t.start()
