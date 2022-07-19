#include <Python.h>
#include <cuda_runtime_api.h>
#include <cuda.h>


typedef struct {
    PyObject_HEAD
    void* data;
    Py_ssize_t size;
} PinnedBuffer;


static int
PinnedBuffer_init(PinnedBuffer *self, PyObject *args, PyObject *kwds)
{
    // init may have already been called
    if (self->data != NULL)
        cudaFreeHost(self->data);

    int size = 0;
    static char *kwlist[] = {"size", NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &size))
        return -1;

    cudaError_t result = cudaMallocHost(&(self->data), size);
    if (result != cudaSuccess)
    {
        PyErr_SetString(PyExc_RuntimeError, cudaGetErrorString(result));
        return -1;
    }

    self->size = size;

    return 0;
}


/* this function is called when the object is deallocated */
static void
PinnedBuffer_dealloc(PinnedBuffer* self)
{
    if (self->data != NULL)
        cudaFreeHost(self->data);

    Py_TYPE(self)->tp_free((PyObject*)self);
}


/* Here is the buffer interface function */
static int
PinnedBuffer_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
  if (view == NULL) {
    PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
    return -1;
  }

  PinnedBuffer* self = (PinnedBuffer*)obj;
  view->obj = (PyObject*)self;
  view->buf = (void*)(self->data);
  view->len = self->size;
  view->readonly = 0;
  view->itemsize = sizeof(uint8_t);
  view->format = "b";  // integer
  view->ndim = 1;
  view->shape = &(self->size);  // length-1 sequence of dimensions
  view->strides = &(view->itemsize);  // for the simple case we can do this
  view->suboffsets = NULL;
  view->internal = NULL;

  Py_INCREF(self);  // need to increase the reference count
  return 0;
}


static PyBufferProcs PinnedBuffer_as_buffer = {
  (getbufferproc)PinnedBuffer_getbuffer,
  (releasebufferproc)0,
};


static PyTypeObject PinnedBufferType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "rpcdataloader.pinned_buffer.PinnedBuffer",
    .tp_doc = PyDoc_STR("Cuda host memory array with buffer interface."),
    .tp_basicsize = sizeof(PinnedBuffer),
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor)PinnedBuffer_dealloc,
    .tp_init = (initproc)PinnedBuffer_init,
    .tp_as_buffer = &PinnedBuffer_as_buffer
};


static PyModuleDef pinned_buffer_module = {
    PyModuleDef_HEAD_INIT,
    "pinned_buffer",
    "Extension type for myarray object.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};


PyMODINIT_FUNC
PyInit_pinned_buffer(void)
{
    PyObject* m;

    PinnedBufferType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PinnedBufferType) < 0)
        return NULL;

    m = PyModule_Create(&pinned_buffer_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PinnedBufferType);
    PyModule_AddObject(m, "PinnedBuffer", (PyObject *)&PinnedBufferType);
    return m;
}