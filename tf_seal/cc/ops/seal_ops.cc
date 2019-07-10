#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("SealKeyGen")
    .Output("pub_key: variant")
    .Output("sec_key: variant")
    .SetIsStateful();

REGISTER_OP("SealEncrypt")
    .Attr("dtype: {float32, float64}")
    .Input("in: dtype")
    .Input("key: variant")
    .Output("val: variant")
    .SetIsStateful();

REGISTER_OP("SealDecrypt")
    .Attr("dtype: {float, double}")
    .Input("val: variant")
    .Input("key: variant")
    .Output("out: dtype")
    .SetIsStateful();

REGISTER_OP("SealEncode")
    .Attr("dtype: {float32, float64}")
    .Input("in: dtype")
    .Output("val: variant")
    .SetIsStateful();

REGISTER_OP("SealDecode")
    .Attr("dtype: {float, double}")
    .Input("val: variant")
    .Output("out: dtype")
    .SetIsStateful();

REGISTER_OP("SealAdd")
    .Input("a: variant")
    .Input("b: variant")
    .Output("out: variant")
    .SetIsStateful();

REGISTER_OP("SealAddPlain")
    .Input("a: variant")
    .Input("b: variant")
    .Output("out: variant")
    .SetIsStateful();
