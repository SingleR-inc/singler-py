#include "def.h"
#include "utils.h"

#include "singlepp/singlepp.hpp"
#include "tatami/tatami.hpp"
#include "pybind11/pybind11.h"

#include <vector>
#include <cstdint>
#include <stdexcept>

pybind11::tuple classify_single(const MatrixPointer& test, const TrainedSingleIntersectPointer& built, double quantile, bool use_fine_tune, double fine_tune_threshold, int nthreads) {
    // Setting up outputs.
    size_t ncells = test->ncol();
    pybind11::array_t<MatrixIndex> best(ncells);
    pybind11::array_t<MatrixValue> delta(ncells);

    singlepp::ClassifySingleBuffers<MatrixIndex, MatrixValue> buffers;
    buffers.best = static_cast<MatrixIndex*>(best.request().ptr);
    buffers.delta = static_cast<MatrixValue*>(delta.request().ptr);

    size_t nlabels = built->num_labels();
    pybind11::array_t<MatrixValue, pybind11::array::f_style> scores({ ncells, nlabels });
    buffers.scores.resize(nlabels);
    auto sptr = static_cast<MatrixValue*>(scores.request().ptr);
    for (size_t l = 0; l < nlabels; ++l) {
        buffers.scores[l] = sptr + ncells * l; // already size_t, no need to cast.
    }

    // Running the analysis.
    singlepp::ClassifySingleOptions opts;
    opts.num_threads = nthreads;
    opts.quantile = quantile;
    opts.fine_tune = use_fine_tune;
    opts.fine_tune_threshold = fine_tune_threshold;
    singlepp::classify_single_intersect(*test, *built, buffers, opts);

    pybind11::tuple output(3);
    output[0] = best;
    output[1] = scores;
    output[2] = delta;
    return output;
}

void init_classify_single(pybind11::module& m) {
    m.def("classify_single", &classify_single);
}
