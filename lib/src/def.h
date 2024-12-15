#ifndef DEF_H
#define DEF_H

#include <cstdint>
#include <memory>
#include "singlepp/singlepp.hpp"
#include "mattress.h"

typedef singlepp::TrainedSingleIntersect<mattress::MatrixIndex, mattress::MatrixValue> TrainedSingleIntersect;
typedef std::shared_ptr<TrainedSingleIntersect> TrainedSingleIntersectPointer;
typedef singlepp::TrainedIntegrated<mattress::MatrixIndex> TrainedIntegrated;
typedef std::shared_ptr<TrainedIntegrated> TrainedIntegratedPointer;

#endif
