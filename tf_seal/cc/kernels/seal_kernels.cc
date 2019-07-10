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

#include "tf_seal/cc/kernels/seal_tensors.h"

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
  const SealTensor* t = input.scalar<Variant>()().get<SealTensor>();
  if (t == nullptr) {
    return InvalidArgument("Input handle is not a seal tensor. Saw: '",
                                   input.scalar<Variant>()().DebugString(),
                                   "'");
  }

  *res = t;
  return Status::OK();
}

Status GetCipherTensor(OpKernelContext* ctx, int index, const CipherTensor** res) {
  const Tensor& input = ctx->input(index);

  // TODO(justin1121): check scalar type
  const CipherTensor* t = input.scalar<Variant>()().get<CipherTensor>();
  if (t == nullptr) {
    return InvalidArgument("Input handle is not a cipher tensor. Saw: '",
                                   input.scalar<Variant>()().DebugString(),
                                   "'");
  }

  *res = t;
  return Status::OK();
}

Status GetPlainTensor(OpKernelContext* ctx, int index, const PlainTensor** res) {
  const Tensor& input = ctx->input(index);

  // TODO(justin1121): check scalar type
  const PlainTensor* t = input.scalar<Variant>()().get<PlainTensor>();
  if (t == nullptr) {
    return InvalidArgument("Input handle is not a plain tensor. Saw: '",
                                   input.scalar<Variant>()().DebugString(),
                                   "'");
  }

  *res = t;
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

    CipherTensor cipher(input.dim_size(0), input.dim_size(1));

    seal::KeyGenerator keygen(context);
    cipher.pub_key = keygen.public_key();
    cipher.sec_key = keygen.secret_key();

    seal::Encryptor encryptor(context, cipher.pub_key);
    seal::CKKSEncoder encoder(context);


    auto data = input.flat<T>().data();
    auto size = input.flat<T>().size();

    seal::Plaintext x_plain;

    // SEAL only takes doubles so cast to that
    // TODO(justin1121): can we get rid of this nasty copy?!
    encoder.encode(std::vector<double>(data, data + size), kScale, x_plain);
    encryptor.encrypt(x_plain, cipher.value);

    val->scalar<Variant>()() = std::move(cipher);
  }
};

template <typename T>
class SealDecryptOp : public OpKernel {
 public:
  explicit SealDecryptOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const CipherTensor* val = nullptr;
    OP_REQUIRES_OK(ctx, GetCipherTensor(ctx, 0, &val));

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

template <typename T>
class SealEncodeOp : public OpKernel {
 public:
  explicit SealEncodeOp(OpKernelConstruction* context) : OpKernel(context) {}

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

    PlainTensor plain(input.dim_size(0), input.dim_size(1));

    seal::CKKSEncoder encoder(context);

    auto data = input.flat<T>().data();
    auto size = input.flat<T>().size();

    // SEAL only takes doubles so cast to that
    // TODO(justin1121): can we get rid of this nasty copy?!
    encoder.encode(std::vector<double>(data, data + size), kScale, plain.value);

    val->scalar<Variant>()() = std::move(plain);
  }
};

template <typename T>
class SealDecodeOp : public OpKernel {
 public:
  explicit SealDecodeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const PlainTensor* val = nullptr;
    OP_REQUIRES_OK(ctx, GetPlainTensor(ctx, 0, &val));

    auto output_shape = TensorShape{val->rows(), val->cols()};

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    auto context = SetParams();

    seal::CKKSEncoder encoder(context);

    std::vector<double> result;
    encoder.decode(val->value, result);

    T* data = output->flat<T>().data();
    size_t size = output->flat<T>().size();

    // SEAL only returns doubles so implicitly cast back to T here, e.g. float
    std::copy_n(result.begin(), size, data);
  }
};

class SealAddOp : public OpKernel {
 public:
  explicit SealAddOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const CipherTensor* a = nullptr;
    OP_REQUIRES_OK(ctx, GetCipherTensor(ctx, 0, &a));

    const CipherTensor* b = nullptr;
    OP_REQUIRES_OK(ctx, GetCipherTensor(ctx, 1, &b));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));

    CipherTensor res(a->rows(), b->rows());

    auto context = SetParams();

    seal::Evaluator evaluator(context);

    evaluator.add(a->value, b->value, res.value);

    output->scalar<Variant>()() = std::move(res);
  }
};

class SealAddPlainOp : public OpKernel {
 public:
  explicit SealAddPlainOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const CipherTensor* a = nullptr;
    OP_REQUIRES_OK(ctx, GetCipherTensor(ctx, 0, &a));

    const PlainTensor* b = nullptr;
    OP_REQUIRES_OK(ctx, GetPlainTensor(ctx, 1, &b));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));

    CipherTensor res(a->rows(), b->rows());

    auto context = SetParams();

    seal::Evaluator evaluator(context);

    evaluator.add_plain(a->value, b->value, res.value);

    output->scalar<Variant>()() = std::move(res);
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SealEncrypt").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), \
      SealEncryptOp<T>);                                                 \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SealDecrypt").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), \
      SealDecryptOp<T>);                                                 \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SealEncode").Device(DEVICE_CPU).TypeConstraint<T>("dtype"),  \
      SealEncodeOp<T>);                                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SealDecode").Device(DEVICE_CPU).TypeConstraint<T>("dtype"),  \
      SealDecodeOp<T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

REGISTER_KERNEL_BUILDER(Name("SealAdd").Device(DEVICE_CPU), SealAddOp);
REGISTER_KERNEL_BUILDER(Name("SealAddPlain").Device(DEVICE_CPU), SealAddPlainOp);

}
