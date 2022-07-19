import threading
import time

import pytest

from rpcdataloader.utils import Future

# try:
#     import torch
# except ImportError:
#     pass
# else:
#     pytest.skip(allow_module_level=True)


def wait_and_set(f, error=False):
    time.sleep(1)
    if error:
        f.set_exception(RuntimeError("hello"))
    else:
        f.set_result("hello")


def test_wait():
    f = Future()

    t = threading.Thread(target=wait_and_set, args=[f])
    t.start()

    assert f.wait() == "hello"
    assert f.value() == "hello"

    t.join()


def test_wait_exception():
    f = Future()

    t = threading.Thread(target=wait_and_set, args=[f, True])
    t.start()

    with pytest.raises(RuntimeError):
        f.wait()

    t.join()


def test_then():
    f = Future()

    def then(f_):
        return f_.wait() + "2"

    f2 = f.then(then)

    t = threading.Thread(target=wait_and_set, args=[f])
    t.start()

    assert f2.wait() == "hello2"

    t.join()
