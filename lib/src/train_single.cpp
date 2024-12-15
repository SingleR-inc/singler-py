#include "def.h"
#include "utils.h"
#include "mattress.h"
#include "knncolle_py.h"

#include "singlepp/singlepp.hpp"
#include "tatami/tatami.hpp"
#include "knncolle/knncolle.hpp"
#include "pybind11/pybind11.h"

#include <vector>
#include <memory>

TrainedSingleIntersectPointer train_single(
    const pybind11::array& test_features,
    uintptr_t ref_ptr,
    const pybind11::array& ref_features,
    const pybind11::array& labels,
    const pybind11::list& markers,
    uintptr_t builder_ptr,
    int nthreads)
{
    const auto& ref = mattress::cast(ref_ptr)->ptr;

    singlepp::TrainSingleOptions<knncolle_py::Index, knncolle_py::MatrixValue> opts;
    opts.num_threads = nthreads;
    opts.top = -1; // Use all available markers; assume subsetting was applied on the Python side.

    const auto& builder = knncolle_py::cast_builder(builder_ptr)->ptr;
    typedef std::shared_ptr<knncolle::Builder<knncolle_py::SimpleMatrix, knncolle_py::Distance> > BuilderPointer;
    opts.trainer = BuilderPointer(BuilderPointer{}, builder.get()); // make a no-op shared pointer.

    auto NR = ref->nrow();
    auto NC = ref->ncol();
    if (static_cast<mattress::MatrixIndex>(labels.size()) != NC) {
        throw std::runtime_error("length of 'labels' is equal to the number of columns of 'ref'");
    }

    // Setting up the markers.
    size_t ngroups = markers.size();
    singlepp::Markers<mattress::MatrixIndex> markers2(ngroups);
    for (size_t m = 0; m < ngroups; ++m) {
        auto curmarkers = markers[m].cast<pybind11::list>();
        auto& curmarkers2 = markers2[m];
        size_t inner_ngroups = curmarkers.size();
        curmarkers2.resize(inner_ngroups);

        for (size_t n = 0; n < inner_ngroups; ++n) {
            auto seq = curmarkers[n].cast<pybind11::array>();
            auto sptr = check_numpy_array<mattress::MatrixIndex>(seq);
            auto& seq2 = curmarkers2[n];
            seq2.insert(seq2.end(), sptr, sptr + seq.size());
        }
    }

    // Preparing the features.
    size_t ninter = test_features.size();
    if (ninter != static_cast<size_t>(ref_features.size())) {
        throw std::runtime_error("length of 'test_features' and 'ref_features' should be the same");
    }
    auto tfptr = check_numpy_array<uint32_t>(test_features);
    auto rfptr = check_numpy_array<uint32_t>(ref_features);
    singlepp::Intersection<mattress::MatrixIndex> inter;
    inter.reserve(ninter);
    for (size_t i = 0; i < ninter; ++i) {
        inter.emplace_back(tfptr[i], rfptr[i]);
    }

    // Building the indices.
    auto built = singlepp::train_single_intersect(
        inter,
        *ref,
        check_numpy_array<uint32_t>(labels),
        std::move(markers2),
        opts
    );

    return TrainedSingleIntersectPointer(new decltype(built)(std::move(built)));
}

pybind11::array_t<mattress::MatrixIndex> get_markers_from_single_reference(const TrainedSingleIntersectPointer& ptr) {
    const auto& rsub = ptr->get_ref_subset();
    return pybind11::array_t<mattress::MatrixIndex>(rsub.size(), rsub.data());
}

uint32_t get_num_markers_from_single_reference(const TrainedSingleIntersectPointer& ptr) {
    return ptr->get_ref_subset().size();
}

uint32_t get_num_labels_from_single_reference(const TrainedSingleIntersectPointer& ptr) {
    return ptr->num_labels();
}

void init_train_single(pybind11::module& m) {
    m.def("train_single", &train_single);
    m.def("get_markers_from_single_reference", &get_markers_from_single_reference);
    m.def("get_num_markers_from_single_reference", &get_num_markers_from_single_reference);
    m.def("get_num_labels_from_single_reference", &get_num_labels_from_single_reference);
}
