#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("SealImport")
    .Attr("dtype: {float32, float64}")
    .Input("in: dtype")
    .Output("val: variant")
    .SetIsStateful();