#ifndef DEF_H
#define DEF_H

#include <cstdint>
#include <memory>
#include "tatami/tatami.hpp"
#include "singlepp/singlepp.hpp"

typedef uint32_t MatrixIndex;
typedef double MatrixValue;
typedef std::shared_ptr<tatami::Matrix<MatrixValue, MatrixIndex> > MatrixPointer;

typedef std::shared_ptr<singlepp::TrainedSingleIntersect<MatrixIndex, MatrixValue> > TrainedSingleIntersectPointer;
typedef std::shared_ptr<singlepp::TrainedIntegrated<MatrixIndex> > TrainedIntegratedPointer;

#endif
