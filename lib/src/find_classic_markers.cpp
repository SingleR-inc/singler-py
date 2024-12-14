#include "def.h"
#include "utils.h"
#include "mattress.h"

#include "singlepp/singlepp.hpp"
#include "tatami/tatami.hpp"
#include "pybind11/pybind11.h"

#include <vector>
#include <cstdint>
#include <stdexcept>

pybind11::list find_classic_markers(uint32_t num_labels, uint32_t num_genes, const pybind11::list& reference, const pybind11::list& labels, int de_n, int nthreads) {
    size_t num_ref = reference.size();
    if (num_ref != static_cast<size_t>(labels.size())) {
        throw std::runtime_error("'ref' and 'labels' should have the same length");
    }

    std::vector<const tatami::Matrix<mattress::MatrixValue, mattress::MatrixIndex>*> ref_ptrs;
    ref_ptrs.reserve(num_ref);
    std::vector<const uint32_t*> lab_ptrs;
    lab_ptrs.reserve(num_ref);

    for (size_t r = 0; r < num_ref; ++r) {
        auto ptr = mattress::cast(reference[r].cast<uintptr_t>())->ptr.get();
        ref_ptrs.emplace_back(ptr);
        if (ptr->nrow() != num_genes) {
            throw std::runtime_error("each entry of 'ref' should have number of rows equal to 'ngenes'");
        }

        // No copy, so it's fine to create a pointer and discard the casted array.
        auto lab = labels[r].cast<pybind11::array>();
        if (lab.size() != static_cast<size_t>(ptr->ncol())) {
            throw std::runtime_error("number of columns in each 'ref' should equal the length of the corresponding entry of 'labels'");
        }

        lab_ptrs.push_back(check_numpy_array<uint32_t>(lab));
    }

    singlepp::ChooseClassicMarkersOptions opts;
    opts.number = de_n;
    opts.num_threads = nthreads;
    auto store = singlepp::choose_classic_markers(ref_ptrs, lab_ptrs, opts);

    pybind11::list output(num_labels);
    for (uint32_t l = 0; l < num_labels; ++l) {
        const auto& src = store[l];
        pybind11::list dest(num_labels);
        for (uint32_t l2 = 0; l2 < num_labels; ++l2) {
            dest[l2] = pybind11::array_t<mattress::MatrixIndex>(src[l2].size(), src[l2].data());
        }
        output[l] = dest;
    }

    return output;
}

uint32_t number_of_classic_markers(uint32_t num_labels) {
    return singlepp::number_of_classic_markers(num_labels);
}

void init_find_classic_markers(pybind11::module& m) {
    m.def("find_classic_markers", &find_classic_markers);
    m.def("number_of_classic_markers", &number_of_classic_markers);
}
