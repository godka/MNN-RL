//
//  ConvolutionCommon.hpp
//  MNN
//
//  Created by MNN on 2020/03/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionCommon_hpp
#define ConvolutionCommon_hpp
#include "AutoStorage.h"
#include "Execution.hpp"
#include "MNN_generated.h"
namespace MNN {
class MNN_PUBLIC ConvolutionCommon : public Execution {
public:
    struct Int8Common {
        AutoStorage<int8_t> weight;
        AutoStorage<float> alpha;
        AutoStorage<float> weightFloat;
        const IDSTQuan* quan;
    };
    static std::shared_ptr<Int8Common> load(const IDSTQuan* quan, bool forceFloat = false);
    static void getConvParameters(std::shared_ptr<ConvolutionCommon::Int8Common> *quanCommon, const MNN::Convolution2D *conv2d, const float** originWeight, int* originWeightSize);

    // Return padX, padY
    static std::pair<int, int> convolutionPad(const Tensor* input, const Tensor* output,
                                              const Convolution2DCommon* common);
    static std::pair<int, int> convolutionTransposePad(const Tensor* input, const Tensor* output,
                                                       const Convolution2DCommon* common);
    struct Im2ColParameter {
        int32_t padX;
        int32_t padY;
        int32_t dilateX;
        int32_t dilateY;
        int32_t strideX;
        int32_t strideY;
        int32_t kernelX;
        int32_t kernelY;
        int32_t icDiv4;
        int32_t kernelCountUnit;
        int32_t iw;
        int32_t ih;
        int32_t ow;
        int32_t oh;
    };
};
} // namespace MNN
#endif