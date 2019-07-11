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

#include "tf_seal/cc/kernels/key_variants.h"
#include "tf_seal/cc/kernels/seal_tensors.h"

namespace tf_seal {

using tensorflow::DEVICE_CPU;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TensorShapeUtils;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;

const double kScale = pow(2.0, 40);
const size_t kPolyModulusDegree = 8192;

template <typename T>
Status GetVariant(OpKernelContext* ctx, int index, const T** res) {
  const Tensor& input = ctx->input(index);

  // TODO(justin1121): check scalar type
  const T* t = input.scalar<Variant>()().get<T>();
  if (t == nullptr) {
    return InvalidArgument("Input handle is not a cipher tensor. Saw: '",
                           input.scalar<Variant>()().DebugString(), "'");
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

void ModSwitchIfNeeded(std::shared_ptr<seal::SEALContext> context,
                     std::shared_ptr<seal::Evaluator> evaluator, const Ciphertext& a,
                     const Ciphertext& to_rescale,
                     Ciphertext& dest) {
  auto a_index = context->get_context_data(a.parms_id())->chain_index();
  auto rescale_index = context->get_context_data(to_rescale.parms_id())->chain_index();

  if(a_index < rescale_index) {
    evaluator->mod_switch_to(to_rescale, a.parms_id(), dest);
  } else {
    dest = to_rescale;
  }
}

class SealKeyGenOp : public OpKernel {
 public:
  explicit SealKeyGenOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* out0;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &out0));

    Tensor* out1;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape{}, &out1));

    Tensor* out2;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape{}, &out2));

    auto context = SetParams();

    seal::KeyGenerator gen(context);

    out0->scalar<Variant>()() = PublicKeyVariant(gen.public_key());
    out1->scalar<Variant>()() = SecretKeyVariant(gen.secret_key());
    out2->scalar<Variant>()() = RelinKeyVariant(gen.relin_keys());
  }
};

template <typename T>
class SealEncryptOp : public OpKernel {
 public:
  explicit SealEncryptOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(input.shape()),
        InvalidArgument("value expected to be a matrix ",
                        "but got shape: ", input.shape().DebugString()));

    // TODO(justin1121): this only temporary until we figure out the best way to
    // encode large number of elements: the encoder can only handle
    // kPolyModulusDegree / 2 at a time or maybe in total for each encoder?
    // something to figure out
    OP_REQUIRES(
        ctx, input.NumElements() <= kPolyModulusDegree / 2,
        InvalidArgument("too many elements, must be less than or equal to ",
                        kPolyModulusDegree / 2));

    const PublicKeyVariant* key = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 1, &key));

    Tensor* val;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &val));

    auto context = SetParams();

    CipherTensor cipher(input.dim_size(0), input.dim_size(1));

    seal::Encryptor encryptor(context, key->key);
    seal::CKKSEncoder encoder(context);

    auto data = input.flat<T>().data();
    auto size = input.flat<T>().size();

    seal::Plaintext plain;

    // SEAL only takes doubles so cast to that
    // TODO(justin1121): can we get rid of this nasty copy?!
    encoder.encode(std::vector<double>(data, data + size), kScale, plain);
    encryptor.encrypt(plain, cipher.value);

    val->scalar<Variant>()() = std::move(cipher);
  }
};

template <typename T>
class SealDecryptOp : public OpKernel {
 public:
  explicit SealDecryptOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const CipherTensor* val = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 0, &val));

    const SecretKeyVariant* key = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 1, &key));

    auto output_shape = TensorShape{val->rows(), val->cols()};

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    auto context = SetParams();

    seal::Decryptor decryptor(context, key->key);

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

class SealAddOp : public OpKernel {
 public:
  explicit SealAddOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const CipherTensor* a = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 0, &a));

    const CipherTensor* b = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 1, &b));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));

    auto context = SetParams();

    CipherTensor res(*a);

    auto evaluator = std::make_shared<seal::Evaluator>(context);

    Ciphertext new_b;
    ModSwitchIfNeeded(context, evaluator, a->value, b->value, new_b);

    Ciphertext new_a;
    ModSwitchIfNeeded(context, evaluator,  new_b, a->value, new_a);

    evaluator->add(new_a, new_b, res.value);

    output->scalar<Variant>()() = std::move(res);
  }
};

template <typename T>
class SealAddPlainOp : public OpKernel {
 public:
  explicit SealAddPlainOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const CipherTensor* a = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 0, &a));
    const Tensor& b = ctx->input(1);

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));

    auto context = SetParams();

    seal::CKKSEncoder encoder(context);

    auto b_data = b.flat<T>().data();
    auto b_size = b.flat<T>().size();
    seal::Plaintext plain;
    encoder.encode(std::vector<double>(b_data, b_data + b_size), kScale, plain);

    CipherTensor res(*a);

    seal::Evaluator evaluator(context);

    evaluator.mod_switch_to_inplace(plain, a->value.parms_id());

    evaluator.add_plain(a->value, plain, res.value);

    output->scalar<Variant>()() = std::move(res);
  }
};

class SealMulOp : public OpKernel {
 public:
  explicit SealMulOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const CipherTensor* a = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 0, &a));

    const CipherTensor* b = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 1, &b));

    const RelinKeyVariant* relin_key = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 2, &relin_key));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));

    auto context = SetParams();

    CipherTensor res(*a);

    auto evaluator = std::make_shared<seal::Evaluator>(context);

    Ciphertext new_b;
    ModSwitchIfNeeded(context, evaluator, a->value, b->value, new_b);

    Ciphertext new_a;
    ModSwitchIfNeeded(context, evaluator,  new_b, a->value, new_a);

    evaluator->multiply(new_a, new_b, res.value);
    evaluator->relinearize_inplace(res.value, relin_key->key);
    evaluator->rescale_to_next_inplace(res.value);

    output->scalar<Variant>()() = std::move(res);
  }
};

template <typename T>
class SealMulPlainOp : public OpKernel {
 public:
  explicit SealMulPlainOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const CipherTensor* a = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 0, &a));

    const Tensor& b = ctx->input(1);

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));

    auto context = SetParams();

    seal::CKKSEncoder encoder(context);

    auto b_data = b.flat<T>().data();
    auto b_size = b.flat<T>().size();
    seal::Plaintext plain;
    encoder.encode(std::vector<double>(b_data, b_data + b_size), kScale, plain);

    CipherTensor res(*a);

    seal::Evaluator evaluator(context);

    evaluator.mod_switch_to_inplace(plain, a->value.parms_id());

    evaluator.multiply_plain(a->value, plain, res.value);
    evaluator.rescale_to_next_inplace(res.value);

    output->scalar<Variant>()() = std::move(res);
  }
};

// Register the CPU kernels.
#define REGISTER_GENERIC_OPS(T)                                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SealEncrypt").Device(DEVICE_CPU).TypeConstraint<T>("dtype"),  \
      SealEncryptOp<T>);                                                  \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SealDecrypt").Device(DEVICE_CPU).TypeConstraint<T>("dtype"),  \
      SealDecryptOp<T>);                                                  \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SealAddPlain").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), \
      SealAddPlainOp<T>);                                                 \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SealMulPlain").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), \
      SealMulPlainOp<T>);

REGISTER_GENERIC_OPS(float);
REGISTER_GENERIC_OPS(double);

REGISTER_KERNEL_BUILDER(Name("SealKeyGen").Device(DEVICE_CPU), SealKeyGenOp);
REGISTER_KERNEL_BUILDER(Name("SealAdd").Device(DEVICE_CPU), SealAddOp);
REGISTER_KERNEL_BUILDER(Name("SealMul").Device(DEVICE_CPU), SealMulOp);


}  // namespace tf_seal
