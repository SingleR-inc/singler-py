#ifndef DEF_H
#define DEF_H

#include <cstdint>
#include <memory>
#include "tatami/tatami.hpp"
#include "singlepp/singlepp.hpp"

typedef uint32_t MatrixIndex;
typedef double MatrixValue;
typedef std::shared_ptr<tatami::Matrix<MatrixValue, MatrixIndex> > MatrixPointer;

typedef std::shared_ptr<knncolle::Builder<knncolle::SimpleMatrix<uint32_t, uint32_t, double>, double> > BuilderPointer;

typedef singlepp::TrainedSingleIntersect<MatrixIndex, MatrixValue> TrainedSingleIntersect;
typedef std::shared_ptr<TrainedSingleIntersect> TrainedSingleIntersectPointer;
typedef singlepp::TrainedIntegrated<MatrixIndex> TrainedIntegrated;
typedef std::shared_ptr<TrainedIntegrated> TrainedIntegratedPointer;

#endif
