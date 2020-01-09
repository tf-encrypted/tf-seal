#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("SealKeyGen")
    .Attr("gen_public: bool = True")
    .Attr("gen_relin: bool = False")
    .Attr("gen_galois: bool = False")
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
    .Input("pub_key: variant")
    .Output("out: variant")
    .SetIsStateful();

REGISTER_OP("SealMulPlain")
    .Attr("dtype: {float32, float64}")
    .Input("a: variant")
    .Input("b: dtype")
    .Output("out: variant")
    .SetIsStateful();

REGISTER_OP("SealMatMul")
    .Input("a: variant")
    .Input("b: variant")
    .Input("pub_key: variant")
    .Output("out: variant")
    .SetIsStateful();

REGISTER_OP("SealMatMulPlain")
    .Attr("dtype: {float32, float64}")
    .Input("a: variant")
    .Input("b: dtype")
    .Input("pub_key: variant")
    .Output("out: variant")
    .SetIsStateful();

REGISTER_OP("SealPolyEval")
    .Attr("dtype: {float32, float64}")
    .Input("x: variant")
    .Input("coeffs: dtype")
    .Input("pub_key: variant")
    .Output("out: variant")
    .SetIsStateful();

REGISTER_OP("SealSavePublickey")
    .Input("filename: string")
    .Input("pub_key: variant")    
    // .Output("out:variant")
    .SetIsStateful();

REGISTER_OP("SealLoadPublickey")
    .Input("a: variant")
    .Input("filename: string")
    .SetIsStateful();

REGISTER_OP("SealSaveSecretkey")
    .Input("filename: string")
    .Input("secretkey: variant")
    // .Output("out:variant")
    .SetIsStateful();

REGISTER_OP("SealLoadSecretkey")
    .Input("a: variant")
    .Input("filename: string")
    .SetIsStateful();

REGISTER_OP("SaveCipherText")
    .Input("filename: string")
    .Input("a: variant")
    .SetIsStateful();

REGISTER_OP("LoadCipherText")
    .Input("a: variant")
    .Input("filename: string")
    .SetIsStateful();