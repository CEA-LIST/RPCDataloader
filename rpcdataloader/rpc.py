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
import select
import socket
import struct
import sys
import threading
import time
import weakref
from typing import Any, Callable, Dict, TypeVar

if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle

import torch
from tblib import pickling_support

# absolute import required for unpickling
from rpcdataloader.utils import pkl_dispatch_table

pickling_support.install()


_T = TypeVar("_T")


def _serialize(obj, buffer_cb=None):
    buffer = io.BytesIO()
    pickler = pickle.Pickler(
        buffer, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=buffer_cb
    )
    pickler.dispatch_table = pkl_dispatch_table
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


def _create_connection(host, timeout, *kargs, **kwargs):
    host, port = host.split(":")
    port = int(port)

    for i in range(int(timeout)):
        try:
            return socket.create_connection(
                (host, port), *kargs, timeout=timeout, **kwargs)
        except OSError as e:
            if i + 1 == timeout:
                raise e from None
            else:
                time.sleep(1)


tls = threading.local()


def _rpc_send_command(host, fut, func, args, kwargs, pin_memory, rref, timeout):
    tls.host = host

    try:
        payload = _serialize((func, args, kwargs, rref))

        # connect to server
        with _create_connection(host, timeout=timeout) as s:
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

                if pin_memory:
                    b = torch.empty(
                        n, dtype=torch.uint8, pin_memory=pin_memory
                    ).numpy()
                else:
                    b = bytearray(n)

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
    func: Callable[..., _T],
    args=None,
    kwargs=None,
    pin_memory=False,
    rref: bool = False,
    timeout=120.0,
) -> torch.futures.Future[_T]:
    """Execute function on remote worker and return the result as a future.

    :param host:
        rpc worker host
    :param func:
        function to execute
    :param args:
        positional arguments
    :param kwargs:
        keword arguments
    :param pin_memory:
        wether buffers (ie: tensors) should be allocated in pinned memory.
    :param rref:
        whether to return the output as a remote reference.
    :param timeout:
        timeout in seconds on network operations

    :return:
        A future that will contain the function return value.

    .. note::
        :attr:`func` and its arguments must be serializable, which exludes
        the usage of lambdas or locally defined functions.
    """
    fut = torch.futures.Future()
    t = threading.Thread(
        target=_rpc_send_command,
        args=(host, fut, func, args, kwargs, pin_memory, rref, timeout),
    )
    t.start()

    return fut


def _handle_client(sock, parallel_sem):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CORK, 1)

    payload = _sock_read(sock, struct.calcsize("L"))
    (n,) = struct.unpack("L", payload)
    payload = _sock_read(sock, n)
    cmd, args, kwargs, rref = pickle.loads(payload)

    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    try:
        with parallel_sem:
            out = cmd(*args, **kwargs)
            if rref:
                out = RRef(obj=out)
        err = None
    except Exception as e:
        out = None
        err = e

    try:
        buffers = []
        payload = _serialize((out, err), buffer_cb=buffers.append)
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


def _create_server(address, *, family=socket.AF_INET):
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind(address)
        sock.listen()
        return sock
    except BaseException as e:
        sock.close()
        raise e from None


def run_worker(host: str, port: int, timeout: float = 120, parallel: int = 1):
    """Start listening and processing remote procedure calls.

    :param host: interface to bind to (set to '0.0.0.0' for all interfaces)
    :param port: port to bind to
    :param timeout: timeout on network transfers from/to client
    :param parallel: max number procedures executing concurrently

    .. warning::
       The workers neither implement authentication nor encryption, any
       user on the network can send arbitrary commands or may listen to the
       traffic from/to the worker.

    .. note::
      - each request is processed in a separate thread
      - network transfers may overlap regardless of :attr:`parallel` argument.
    """
    torch.set_num_threads(1)  # prevent thread competition

    parallel_sem = threading.Semaphore(parallel)
    with _create_server((host, port), family=socket.AF_INET) as sock:
        while True:
            client_sock, _ = sock.accept()
            client_sock.settimeout(timeout)
            t = threading.Thread(
                target=_handle_client, args=[client_sock, parallel_sem]
            )
            t.start()


_handles: Dict[int, Any] = {}


class RRef:
    def __init__(self, obj=None, uid=None):
        if uid is None:
            self.obj = obj
            self.uid = None

        else:
            self.uid = uid
            self.host = tls.host

            weakref.finalize(self, rpc_async, self.host, _handles.pop, [uid])

    @staticmethod
    def wrap(func, args, kwargs):
        return RRef(obj=func(*args, **kwargs))

    @staticmethod
    def _rebuild_remote(uid):
        return RRef(uid=uid)

    @staticmethod
    def _rebuild_local(uid):
        return _handles[uid]

    def __reduce__(self):
        if self.uid is not None:
            return RRef._rebuild_local, (self.uid,)

        else:
            uid = id(self.obj)
            if uid in _handles:
                raise RuntimeError("Only one rref can exist for a given object")

            _handles[uid] = self.obj

            return RRef._rebuild_remote, (uid,)
