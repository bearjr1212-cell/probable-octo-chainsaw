/*
 * Python bindings for the adaptive-precision predicates.
 *
 * The arithmetic itself lives in predicates_impl.h so that cdt.c can call
 * it directly, inlined, without paying a Python round trip per sign test --
 * a triangulation performs millions of them, and the boundary crossing
 * costs far more than the predicate does.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "predicates_impl.h"

static PyObject *py_orient2d(PyObject *self, PyObject *args)
{
    REAL pa[2], pb[2], pc[2];
    (void)self;
    if (!PyArg_ParseTuple(args, "(dd)(dd)(dd)", &pa[0], &pa[1], &pb[0], &pb[1], &pc[0], &pc[1])) {
        return NULL;
    }
    return PyLong_FromLong(sign_of_real(orient2d_value(pa, pb, pc)));
}

/* Exact in-circle sign; see incircle_sign() in predicates_impl.h. */
static PyObject *py_incircle(PyObject *self, PyObject *args)
{
    REAL pa[2], pb[2], pc[2], pd[2];

    (void)self;
    if (!PyArg_ParseTuple(args, "(dd)(dd)(dd)(dd)", &pa[0], &pa[1], &pb[0], &pb[1], &pc[0],
                          &pc[1], &pd[0], &pd[1])) {
        return NULL;
    }

    return PyLong_FromLong(incircle_sign(pa, pb, pc, pd));
}

static PyObject *py_stats(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    return Py_BuildValue("{s:K,s:K,s:K,s:K}", "orient2d_calls", stat_orient2d_calls,
                         "orient2d_adaptive", stat_orient2d_adapt, "incircle_calls",
                         stat_incircle_calls, "incircle_deferred", stat_incircle_defer);
}

static PyObject *py_reset_stats(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    stat_orient2d_calls = 0;
    stat_orient2d_adapt = 0;
    stat_incircle_calls = 0;
    stat_incircle_defer = 0;
    Py_RETURN_NONE;
}

static PyObject *py_machine_constants(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    return Py_BuildValue("{s:d,s:d,s:d}", "epsilon", epsilon, "splitter", splitter,
                         "orient2d_filter", ccwerrboundA);
}

static PyMethodDef PredicateMethods[] = {
    {"orient2d", py_orient2d, METH_VARARGS,
     "orient2d(a, b, c) -> int: exact sign of the orientation determinant."},
    {"incircle", py_incircle, METH_VARARGS,
     "incircle(a, b, c, d) -> int: exact sign of the in-circle determinant."},
    {"stats", py_stats, METH_NOARGS, "Predicate call/fallback counters."},
    {"reset_stats", py_reset_stats, METH_NOARGS, "Zero the counters."},
    {"machine_constants", py_machine_constants, METH_NOARGS,
     "The FP constants derived at import time."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef predicatesmodule = {
    PyModuleDef_HEAD_INIT, "blueprint23d._native.predicates",
    "Adaptive-precision geometric predicates (Shewchuk).", -1, PredicateMethods,
    NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_predicates(void)
{
    exactinit();
    return PyModule_Create(&predicatesmodule);
}
