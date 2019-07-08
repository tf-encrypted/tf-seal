#include <iostream>

#include <seal/seal.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/bounds_check.h"

using namespace tensorflow;

using seal::SEALContext;

std::shared_ptr<SEALContext> set_params() {
    seal::EncryptionParameters parms(seal::scheme_type::BFV);

    size_t poly_modulus_degree = 4096;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(256);
    return SEALContext::Create(parms);
}

template <typename T>
class SealImportOp : public OpKernel {
 public:
  explicit SealImportOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(input.shape()),
                errors::InvalidArgument(
                    "value expected to be a matrix ",
                    "but got shape: ", input.shape().DebugString()));

    Tensor* val;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &val));

    auto context = set_params();

    std::cout << context << std::endl;
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SealImport").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), \
      SealImportOp<T>);                                                  \

REGISTER_CPU(float);
REGISTER_CPU(double);