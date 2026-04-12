#ifndef PYRAMID_PYRAMID_TYPES_H_
#define PYRAMID_PYRAMID_TYPES_H_

namespace pyramid {

// Parameters controlling Laplacian pyramid construction
struct PyramidParams {
    // Number of Laplacian levels to build (exclusive of the Gaussian top)
    int num_levels = 6;
};

}  // namespace pyramid

#endif  // PYRAMID_PYRAMID_TYPES_H_
