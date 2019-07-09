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

using namespace tensorflow;  // NOLINT
using namespace seal;        // NOLINT

using tf_seal::SealTensor;

const double SCALE = pow(2.0, 40);
const size_t POLY_MODULUS_DEGREE = 8192;

Status GetSealTensor(OpKernelContext* ctx, int index, const SealTensor** res) {
  const Tensor& input = ctx->input(index);

  // TODO(justin1121): check scalar type
  const SealTensor* big = input.scalar<Variant>()().get<SealTensor>();
  if (big == nullptr) {
    return errors::InvalidArgument("Input handle is not a seal tensor. Saw: '",
                                   input.scalar<Variant>()().DebugString(),
                                   "'");
  }

  *res = big;
  return Status::OK();
}

std::shared_ptr<SEALContext> set_params() {
  EncryptionParameters parms(scheme_type::CKKS);

  parms.set_poly_modulus_degree(POLY_MODULUS_DEGREE);
  parms.set_coeff_modulus(
      CoeffModulus::Create(POLY_MODULUS_DEGREE, {60, 40, 40, 60}));

  return SEALContext::Create(parms);
}

template <typename T>
class SealEncryptOp : public OpKernel {
 public:
  explicit SealEncryptOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(input.shape()),
                errors::InvalidArgument(
                    "value expected to be a matrix ",
                    "but got shape: ", input.shape().DebugString()));

    // TODO(justin1121): this only temporary until we figure out the best way to
    // encode large number of elements: the encoder can only handle
    // POLY_MODULUS_DEGREE / 2 at a time or maybe in total for each encoder?
    // something to figure out
    OP_REQUIRES(ctx, input.NumElements() <= POLY_MODULUS_DEGREE / 2,
                errors::InvalidArgument(
                    "too many elements, must be less than or equal to ",
                    POLY_MODULUS_DEGREE / 2));

    Tensor* val;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &val));

    auto context = set_params();

    tf_seal::SealTensor s_tensor(input.dim_size(0), input.dim_size(1));

    KeyGenerator keygen(context);
    s_tensor.pub_key = keygen.public_key();
    s_tensor.sec_key = keygen.secret_key();
    Encryptor encryptor(context, s_tensor.pub_key);

    CKKSEncoder encoder(context);

    Plaintext x_plain;

    auto data = input.flat<T>().data();
    size_t size = input.flat<T>().size();

    // SEAL only takes doubles so cast to that
    // TODO(justin1121): can we get rid of this nasty copy?!
    encoder.encode(std::vector<double>(data, data + size), SCALE, x_plain);
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

    auto context = set_params();

    Decryptor decryptor(context, val->sec_key);

    Plaintext plain_result;
    decryptor.decrypt(val->value, plain_result);

    CKKSEncoder encoder(context);

    std::vector<double> result;
    encoder.decode(plain_result, result);

    // SEAL only returns doubles so cast back to T here, i.e. float
    T* data = output->flat<T>().data();
    size_t size = output->flat<T>().size();
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
