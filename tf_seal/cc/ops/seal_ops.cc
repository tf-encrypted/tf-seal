#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("SealKeyGen")
    .Output("pub_key: variant")
    .Output("sec_key: variant")
    .Output("relin_key: variant")
    .SetIsStateful();

REGISTER_OP("SealEncrypt")
    .Attr("dtype: {float32, float64}")
    .Input("in: dtype")
    .Input("key: variant")
    .Output("val: variant")
    .SetIsStateful();

REGISTER_OP("SealDecrypt")
    .Attr("dtype: {float32, float64}")
    .Input("val: variant")
    .Input("key: variant")
    .Output("out: dtype")
    .SetIsStateful();

REGISTER_OP("SealAdd")
    .Input("a: variant")
    .Input("b: variant")
    .Output("out: variant")
    .SetIsStateful();

REGISTER_OP("SealAddPlain")
    .Attr("dtype: {float32, float64}")
    .Input("a: variant")
    .Input("b: dtype")
    .Output("out: variant")
    .SetIsStateful();

REGISTER_OP("SealMul")
    .Input("a: variant")
    .Input("b: variant")
    .Input("relin_key: variant")
    .Output("out: variant")
    .SetIsStateful();

REGISTER_OP("SealMulPlain")
    .Attr("dtype: {float32, float64}")
    .Input("a: variant")
    .Input("b: dtype")
    .Output("out: variant")
    .SetIsStateful();
