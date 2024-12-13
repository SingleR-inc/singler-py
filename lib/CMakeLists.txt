cmake_minimum_required(VERSION 3.24)

project(singler
    VERSION 1.0.0
    DESCRIPTION "Building the singler shared library"
    LANGUAGES CXX)

# Defining the targets.
find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(singler
    src/find_classic_markers.cpp
    src/train_single.cpp
    src/classify_single.cpp
    src/train_integrated.cpp
    src/classify_integrated.cpp
    src/init.cpp
)

target_include_directories(singler PRIVATE "${ASSORTHEAD_INCLUDE_DIR}")
target_include_directories(singler PRIVATE "${MATTRESS_INCLUDE_DIR}")
target_include_directories(singler PRIVATE "${KNNCOLLE_INCLUDE_DIR}")

set_property(TARGET singler PROPERTY CXX_STANDARD 17)

target_link_libraries(singler PRIVATE pybind11::pybind11)

set_target_properties(singler PROPERTIES
    OUTPUT_NAME lib_singler
    PREFIX ""
)
