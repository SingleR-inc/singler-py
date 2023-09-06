/* DO NOT MODIFY: this is automatically generated by the cpptypes */

#include <cstring>
#include <stdexcept>
#include <cstdint>

#ifdef _WIN32
#define PYAPI __declspec(dllexport)
#else
#define PYAPI
#endif

static char* copy_error_message(const char* original) {
    auto n = std::strlen(original);
    auto copy = new char[n + 1];
    std::strcpy(copy, original);
    return copy;
}

void* create_markers(int32_t);

void* find_classic_markers(int32_t, const uintptr_t*, const uintptr_t*, int32_t, int32_t);

void free_markers(void*);

void get_markers_for_pair(void*, int32_t, int32_t, int32_t*);

int32_t get_nlabels_from_markers(void*);

int32_t get_nmarkers_for_pair(void*, int32_t, int32_t);

void set_markers_for_pair(void*, int32_t, int32_t, int32_t, const int32_t*);

extern "C" {

PYAPI void free_error_message(char** msg) {
    delete [] *msg;
}

PYAPI void* py_create_markers(int32_t nlabels, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = create_markers(nlabels);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_find_classic_markers(int32_t nref, const uintptr_t* labels, const uintptr_t* ref, int32_t de_n, int32_t nthreads, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = find_classic_markers(nref, labels, ref, de_n, nthreads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_free_markers(void* ptr, int32_t* errcode, char** errmsg) {
    try {
        free_markers(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_get_markers_for_pair(void* ptr, int32_t label1, int32_t label2, int32_t* buffer, int32_t* errcode, char** errmsg) {
    try {
        get_markers_for_pair(ptr, label1, label2, buffer);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI int32_t py_get_nlabels_from_markers(void* ptr, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = get_nlabels_from_markers(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int32_t py_get_nmarkers_for_pair(void* ptr, int32_t label1, int32_t label2, int32_t* errcode, char** errmsg) {
    int32_t output = 0;
    try {
        output = get_nmarkers_for_pair(ptr, label1, label2);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_set_markers_for_pair(void* ptr, int32_t label1, int32_t label2, int32_t n, const int32_t* values, int32_t* errcode, char** errmsg) {
    try {
        set_markers_for_pair(ptr, label1, label2, n, values);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

}