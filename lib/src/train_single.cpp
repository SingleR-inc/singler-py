#include "def.h"
#include "utils.h"

#include "singlepp/singlepp.hpp"
#include "tatami/tatami.hpp"
#include "knncolle/knncolle.hpp"
#include "pybind11/pybind11.h"

#include <vector>
#include <memory>

TrainedSingleIntersectPointer train_single(
    const pybind11::array& test_features,
    const MatrixPointer& ref,
    const pybind11::array& ref_features,
    const pybind11::array& labels,
    const pybind11::list& markers,
    const std::shared_ptr<knncolle::Builder<knncolle::SimpleMatrix<uint32_t, uint32_t, double>, double> >& builder,
    int nthreads)
{
    singlepp::TrainSingleOptions<uint32_t, double> opts;
    opts.num_threads = nthreads;
    opts.top = -1; // Use all available markers; assume subsetting was applied on the R side.

    opts.trainer = builder; // std::shared_ptr<BiocNeighbors::Builder>(std::shared_ptr<BiocNeighbors::Builder>{}, bptr.get()); // make a no-op shared pointer.

    auto NR = ref->nrow();
    auto NC = ref->ncol();
    if (static_cast<MatrixIndex>(labels.size()) != NC) {
        throw std::runtime_error("length of 'labels' is equal to the number of columns of 'ref'");
    }

    // Setting up the markers. We assume that these are already 0-indexed on the R side.
    size_t ngroups = markers.size();
    singlepp::Markers<MatrixIndex> markers2(ngroups);
    for (size_t m = 0; m < ngroups; ++m) {
        auto curmarkers = markers[m].cast<pybind11::list>();
        auto& curmarkers2 = markers2[m];
        size_t inner_ngroups = curmarkers.size();
        curmarkers2.resize(inner_ngroups);

        for (size_t n = 0; n < inner_ngroups; ++n) {
            auto seq = curmarkers[n].cast<pybind11::array>();
            auto sptr = check_numpy_array<MatrixIndex>(seq);
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
    singlepp::Intersection<MatrixIndex> inter;
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

pybind11::array_t<MatrixIndex> get_ref_subset(const TrainedSingleIntersectPointer& ptr) {
    const auto& rsub = ptr->get_ref_subset();
    return pybind11::array_t<MatrixIndex>(rsub.size(), rsub.data());
}

void init_train_single(pybind11::module& m) {
    m.def("train_single", &train_single);
    m.def("get_ref_subset", &get_ref_subset);
}