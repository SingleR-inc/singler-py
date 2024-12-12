#include "def.h"
#include "pybind11/pybind11.h"

void init_find_classic_markers(pybind11::module&);
void init_train_single(pybind11::module&);
void init_classify_single(pybind11::module&);
void init_train_integrated(pybind11::module&);
void init_classify_integrated(pybind11::module&);

PYBIND11_MODULE(lib_singler, m) {
    init_find_classic_markers(m);
    init_train_single(m);
    init_classify_single(m);
    init_train_integrated(m);
    init_classify_integrated(m);

    pybind11::class_<TrainedSingleIntersect, TrainedSingleIntersectPointer>(m, "TrainSingleIntersect");
    pybind11::class_<TrainedIntegrated, TrainedIntegratedPointer>(m, "TrainIntegrated");
}
