#include "def.h"
#include "utils.h"

#include "singlepp/singlepp.hpp"
#include "tatami/tatami.hpp"
#include "pybind11/pybind11.h"

#include <vector>
#include <cstdint>
#include <stdexcept>

pybind11::tuple classify_integrated(
    const MatrixPointer& test, 
    const pybind11::list& results,
    const TrainedIntegratedPointer& integrated_build,
    double quantile,
    bool use_fine_tune,
    double fine_tune_threshold,
    int nthreads)
{
    // Setting up the previous results.
    size_t num_refs = results.size();
    std::vector<const uint32_t*> previous_results;
    previous_results.reserve(num_refs);
    for (size_t r = 0; r < num_refs; ++r) {
        const auto& curres = results[r].cast<pybind11::array>();
        previous_results.push_back(check_numpy_array<uint32_t>(curres));
    }

    // Setting up outputs.
    size_t ncells = test->ncol();
    pybind11::array_t<MatrixIndex> best(ncells);
    pybind11::array_t<MatrixValue> delta(ncells);

    singlepp::ClassifyIntegratedBuffers<MatrixIndex, MatrixValue> buffers;
    buffers.best = static_cast<MatrixIndex*>(best.request().ptr);
    buffers.delta = static_cast<MatrixValue*>(delta.request().ptr);

    pybind11::list scores(num_refs);
    buffers.scores.resize(num_refs);
    for (size_t l = 0; l < num_refs; ++l) {
        scores[l] = pybind11::array_t<MatrixValue>(ncells);
        buffers.scores[l] = static_cast<MatrixValue*>(scores[l].cast<pybind11::array>().request().ptr);
    }

    // Running the integrated scoring.
    singlepp::ClassifyIntegratedOptions<double> opts;
    opts.num_threads = nthreads;
    opts.quantile = quantile;
    opts.fine_tune = use_fine_tune;
    opts.fine_tune_threshold = fine_tune_threshold;
    singlepp::classify_integrated(*test, previous_results, *integrated_build, buffers, opts);

    pybind11::tuple output(3);
    output[0] = best;
    output[1] = scores;
    output[2] = delta;
    return output;
}

void init_classify_integrated(pybind11::module& m) {
    m.def("classify_integrated", &classify_integrated);
}
