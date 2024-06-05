# DO NOT MODIFY: this is automatically generated by the cpptypes

import os
import ctypes as ct
import numpy as np

def _catch_errors(f):
    def wrapper(*args):
        errcode = ct.c_int32(0)
        errmsg = ct.c_char_p(0)
        output = f(*args, ct.byref(errcode), ct.byref(errmsg))
        if errcode.value != 0:
            msg = errmsg.value.decode('ascii')
            lib.free_error_message(errmsg)
            raise RuntimeError(msg)
        return output
    return wrapper

# TODO: surely there's a better way than whatever this is.
dirname = os.path.dirname(os.path.abspath(__file__))
contents = os.listdir(dirname)
lib = None
for x in contents:
    if x.startswith('_core') and not x.endswith("py"):
        lib = ct.CDLL(os.path.join(dirname, x))
        break

if lib is None:
    raise ImportError("failed to find the _core.* module")

lib.free_error_message.argtypes = [ ct.POINTER(ct.c_char_p) ]

def _np2ct(x, expected, contiguous=True):
    if not isinstance(x, np.ndarray):
        raise ValueError('expected a NumPy array')
    if x.dtype != expected:
        raise ValueError('expected a NumPy array of type ' + str(expected) + ', got ' + str(x.dtype))
    if contiguous:
        if not x.flags.c_contiguous and not x.flags.f_contiguous:
            raise ValueError('only contiguous NumPy arrays are supported')
    return x.ctypes.data

lib.py_build_integrated_references.restype = ct.c_void_p
lib.py_build_integrated_references.argtypes = [
    ct.c_int32,
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_build_single_reference.restype = ct.c_void_p
lib.py_build_single_reference.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_classify_integrated_references.restype = None
lib.py_classify_integrated_references.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_classify_single_reference.restype = None
lib.py_classify_single_reference.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.c_uint8,
    ct.c_double,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_create_markers.restype = ct.c_void_p
lib.py_create_markers.argtypes = [
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_find_classic_markers.restype = ct.c_void_p
lib.py_find_classic_markers.argtypes = [
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_integrated_references.restype = None
lib.py_free_integrated_references.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_markers.restype = None
lib.py_free_markers.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_single_reference.restype = None
lib.py_free_single_reference.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_get_markers_for_pair.restype = None
lib.py_get_markers_for_pair.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_int32,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_get_nlabels_from_markers.restype = ct.c_int32
lib.py_get_nlabels_from_markers.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_get_nlabels_from_single_reference.restype = ct.c_int32
lib.py_get_nlabels_from_single_reference.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_get_nmarkers_for_pair.restype = ct.c_int32
lib.py_get_nmarkers_for_pair.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_get_nsubset_from_single_reference.restype = ct.c_int32
lib.py_get_nsubset_from_single_reference.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_get_subset_from_single_reference.restype = None
lib.py_get_subset_from_single_reference.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_number_of_classic_markers.restype = ct.c_int32
lib.py_number_of_classic_markers.argtypes = [
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_set_markers_for_pair.restype = None
lib.py_set_markers_for_pair.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_int32,
    ct.c_int32,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

def build_integrated_references(test_nrow, test_features, nrefs, references, labels, ref_ids, prebuilt, nthreads):
    return _catch_errors(lib.py_build_integrated_references)(test_nrow, _np2ct(test_features, np.int32), nrefs, references, labels, ref_ids, prebuilt, nthreads)

def build_single_reference(ref, labels, markers, approximate, nthreads):
    return _catch_errors(lib.py_build_single_reference)(ref, _np2ct(labels, np.int32), markers, approximate, nthreads)

def classify_integrated_references(mat, assigned, prebuilt, quantile, scores, best, delta, nthreads):
    return _catch_errors(lib.py_classify_integrated_references)(mat, assigned, prebuilt, quantile, scores, _np2ct(best, np.int32), _np2ct(delta, np.float64), nthreads)

def classify_single_reference(mat, subset, prebuilt, quantile, use_fine_tune, fine_tune_threshold, nthreads, scores, best, delta):
    return _catch_errors(lib.py_classify_single_reference)(mat, _np2ct(subset, np.int32), prebuilt, quantile, use_fine_tune, fine_tune_threshold, nthreads, scores, _np2ct(best, np.int32), _np2ct(delta, np.float64))

def create_markers(nlabels):
    return _catch_errors(lib.py_create_markers)(nlabels)

def find_classic_markers(nref, labels, ref, de_n, nthreads):
    return _catch_errors(lib.py_find_classic_markers)(nref, labels, ref, de_n, nthreads)

def free_integrated_references(ptr):
    return _catch_errors(lib.py_free_integrated_references)(ptr)

def free_markers(ptr):
    return _catch_errors(lib.py_free_markers)(ptr)

def free_single_reference(ptr):
    return _catch_errors(lib.py_free_single_reference)(ptr)

def get_markers_for_pair(ptr, label1, label2, buffer):
    return _catch_errors(lib.py_get_markers_for_pair)(ptr, label1, label2, _np2ct(buffer, np.int32))

def get_nlabels_from_markers(ptr):
    return _catch_errors(lib.py_get_nlabels_from_markers)(ptr)

def get_nlabels_from_single_reference(ptr):
    return _catch_errors(lib.py_get_nlabels_from_single_reference)(ptr)

def get_nmarkers_for_pair(ptr, label1, label2):
    return _catch_errors(lib.py_get_nmarkers_for_pair)(ptr, label1, label2)

def get_nsubset_from_single_reference(ptr):
    return _catch_errors(lib.py_get_nsubset_from_single_reference)(ptr)

def get_subset_from_single_reference(ptr, buffer):
    return _catch_errors(lib.py_get_subset_from_single_reference)(ptr, _np2ct(buffer, np.int32))

def number_of_classic_markers(num_labels):
    return _catch_errors(lib.py_number_of_classic_markers)(num_labels)

def set_markers_for_pair(ptr, label1, label2, n, values):
    return _catch_errors(lib.py_set_markers_for_pair)(ptr, label1, label2, n, _np2ct(values, np.int32))