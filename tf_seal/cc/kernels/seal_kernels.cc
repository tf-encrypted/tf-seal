#include <iostream>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

#include "seal/seal.h"

#include "tf_seal/cc/kernels/seal_tensor.h"

namespace tf_seal {

using tensorflow::Status;
using tensorflow::Variant;
using tensorflow::OpKernelContext;
using tensorflow::OpKernelConstruction;
using tensorflow::Tensor;
using tensorflow::errors::InvalidArgument;
using tensorflow::DEVICE_CPU;
using tensorflow::TensorShape;
using tensorflow::OpKernel;
using tensorflow::TensorShapeUtils;

const double kScale = pow(2.0, 40);
const size_t kPolyModulusDegree = 8192;

Status GetSealTensor(OpKernelContext* ctx, int index, const SealTensor** res) {
  const Tensor& input = ctx->input(index);

  // TODO(justin1121): check scalar type
  const SealTensor* big = input.scalar<Variant>()().get<SealTensor>();
  if (big == nullptr) {
    return InvalidArgument("Input handle is not a seal tensor. Saw: '",
                                   input.scalar<Variant>()().DebugString(),
                                   "'");
  }

  *res = big;
  return Status::OK();
}

std::shared_ptr<seal::SEALContext> SetParams() {
  seal::EncryptionParameters parms(seal::scheme_type::CKKS);

  parms.set_poly_modulus_degree(kPolyModulusDegree);
  parms.set_coeff_modulus(
      seal::CoeffModulus::Create(kPolyModulusDegree, {60, 40, 40, 60}));

  return seal::SEALContext::Create(parms);
}

template <typename T>
class SealEncryptOp : public OpKernel {
 public:
  explicit SealEncryptOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(input.shape()),
                InvalidArgument(
                    "value expected to be a matrix ",
                    "but got shape: ", input.shape().DebugString()));

    // TODO(justin1121): this only temporary until we figure out the best way to
    // encode large number of elements: the encoder can only handle
    // kPolyModulusDegree / 2 at a time or maybe in total for each encoder?
    // something to figure out
    OP_REQUIRES(ctx, input.NumElements() <= kPolyModulusDegree / 2,
                InvalidArgument(
                    "too many elements, must be less than or equal to ",
                    kPolyModulusDegree / 2));

    Tensor* val;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &val));

    auto context = SetParams();

    SealTensor s_tensor(input.dim_size(0), input.dim_size(1));

    seal::KeyGenerator keygen(context);
    s_tensor.pub_key = keygen.public_key();
    s_tensor.sec_key = keygen.secret_key();
    seal::Encryptor encryptor(context, s_tensor.pub_key);

    seal::CKKSEncoder encoder(context);

    seal::Plaintext x_plain;

    auto data = input.flat<T>().data();
    size_t size = input.flat<T>().size();

    // SEAL only takes doubles so cast to that
    // TODO(justin1121): can we get rid of this nasty copy?!
    encoder.encode(std::vector<double>(data, data + size), kScale, x_plain);
    encryptor.encrypt(x_plain, s_tensor.value);

    val->scalar<Variant>()() = std::move(s_tensor);
  }
};

template <typename T>
class SealDecryptOp : public OpKernel {
 public:
  explicit SealDecryptOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const SealTensor* val = nullptr;
    OP_REQUIRES_OK(ctx, GetSealTensor(ctx, 0, &val));

    auto output_shape = TensorShape{val->rows(), val->cols()};

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    auto context = SetParams();

    seal::Decryptor decryptor(context, val->sec_key);

    seal::Plaintext plain_result;
    decryptor.decrypt(val->value, plain_result);

    seal::CKKSEncoder encoder(context);

    std::vector<double> result;
    encoder.decode(plain_result, result);

    T* data = output->flat<T>().data();
    size_t size = output->flat<T>().size();

    // SEAL only returns doubles so implicitly cast back to T here, e.g. float
    std::copy_n(result.begin(), size, data);
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SealEncrypt").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), \
      SealEncryptOp<T>);                                                 \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SealDecrypt").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), \
      SealDecryptOp<T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

}
