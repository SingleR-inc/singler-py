#include "def.h"
#include "utils.h"
#include "mattress.h"

#include "singlepp/singlepp.hpp"
#include "tatami/tatami.hpp"
#include "pybind11/pybind11.h"

#include <cstdint>
#include <vector>
#include <memory>

TrainedIntegratedPointer train_integrated(
    const pybind11::list& test_features,
    const pybind11::list& references,
    const pybind11::list& ref_features,
    const pybind11::list& labels,
    const pybind11::list& prebuilt,
    int nthreads) 
{
    size_t nrefs = references.size();
    std::vector<singlepp::TrainIntegratedInput<mattress::MatrixValue, mattress::MatrixIndex, uint32_t> > inputs;
    inputs.reserve(nrefs);
    std::vector<singlepp::Intersection<mattress::MatrixIndex> > intersections(nrefs);

    for (size_t r = 0; r < nrefs; ++r) {
        const auto& curref = mattress::cast(references[r].cast<uintptr_t>())->ptr;
        const auto& curlabels = labels[r].cast<pybind11::array>();
        const auto& curbuilt = prebuilt[r].cast<TrainedSingleIntersectPointer>();

        const auto& test_ids = test_features[r].cast<pybind11::array>();
        const auto& ref_ids = ref_features[r].cast<pybind11::array>();
        size_t ninter = test_ids.size();
        if (ninter != static_cast<size_t>(ref_ids.size())) {
            throw std::runtime_error("length of each entry of 'test_features' and 'ref_features' should be the same");
        }
        const auto* test_ids_ptr = check_numpy_array<mattress::MatrixIndex>(test_ids);
        const auto* ref_ids_ptr = check_numpy_array<mattress::MatrixIndex>(ref_ids);

        auto& curinter = intersections[r];
        curinter.reserve(ninter);
        for (size_t i = 0; i < ninter; ++i) {
            curinter.emplace_back(test_ids_ptr[i], ref_ids_ptr[i]);
        }

        inputs.push_back(singlepp::prepare_integrated_input_intersect(
            curinter,
            *curref,
            check_numpy_array<uint32_t>(curlabels),
            *curbuilt
        ));
    }

    singlepp::TrainIntegratedOptions opts;
    opts.num_threads = nthreads;
    auto finished = singlepp::train_integrated(std::move(inputs), opts);

    return TrainedIntegratedPointer(new TrainedIntegrated(std::move(finished)));
}

void init_train_integrated(pybind11::module& m) {
    m.def("train_integrated", &train_integrated);
}
