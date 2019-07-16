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
#include "tf_seal/cc/kernels/seal_helpers.h"
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

template <typename T>
Status GetVariant(OpKernelContext* ctx, int index, const T** res) {
  const Tensor& input = ctx->input(index);

  // TODO(justin1121): check scalar type
  const T* t = input.scalar<Variant>()().get<T>();
  if (t == nullptr) {
    return InvalidArgument(
        "Input handle is not the correct variant tensor. Saw: '",
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
                       Evaluator* evaluator, const Ciphertext& a,
                       const Ciphertext& to_mod, Ciphertext* dest) {
  auto a_index = context->get_context_data(a.parms_id())->chain_index();
  auto mod_index = context->get_context_data(to_mod.parms_id())->chain_index();

  if (a_index < mod_index) {
    evaluator->mod_switch_to(to_mod, a.parms_id(), *dest);
  } else {
    *dest = to_mod;
  }
}

class SealKeyGenOp : public OpKernel {
 public:
  explicit SealKeyGenOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("gen_public", &gen_public));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("gen_relin", &gen_relin));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("gen_galois", &gen_galois));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* out0;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &out0));

    Tensor* out1;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape{}, &out1));

    auto context = SetParams();

    seal::KeyGenerator gen(context);

    PublicKeysVariant pub_keys;

    if(gen_public) {
      pub_keys.public_key = gen.public_key();
    }

    if(gen_relin) {
      pub_keys.relin_keys = gen.relin_keys();
    }

    if(gen_galois) {
      pub_keys.galois_keys = gen.galois_keys();
    }

    out0->scalar<Variant>()() = std::move(pub_keys);
    out1->scalar<Variant>()() = SecretKeyVariant(gen.secret_key());
  }

  bool gen_public = true;
  bool gen_relin = false;
  bool gen_galois = false;
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

    const PublicKeysVariant* key = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 1, &key));

    Tensor* val;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &val));

    auto context = SetParams();

    CipherTensor cipher(input.dim_size(0), input.dim_size(1));

    seal::Encryptor encryptor(context, key->public_key);
    seal::CKKSEncoder encoder(context);

    auto data = input.flat<T>().data();
    int rows = input.dim_size(0);
    int cols = input.dim_size(1);

    seal::Plaintext plain;

    // SEAL only takes doubles so cast to that
    // TODO(justin1121): can we get rid of this nasty copy?!
    for (int i = 0; i < rows; i++) {
      encoder.encode(
          std::vector<double>(data + (cols * i), data + (cols * (i + 1))),
          kScale, plain);

      encryptor.encrypt(plain, cipher.value[i]);
    }

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
    seal::CKKSEncoder encoder(context);

    seal::Plaintext plain_result;

    auto data = output->flat<T>().data();
    int rows = val->rows();
    int cols = val->cols();

    for (int i = 0; i < rows; i++) {
      decryptor.decrypt(val->value[i], plain_result);

      std::vector<double> result;
      encoder.decode(plain_result, result);

      // SEAL only returns doubles so implicitly cast back to T here, e.g. float
      std::copy_n(result.begin(), cols, data + (cols * i));
    }
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

    Evaluator evaluator(context);

    for (int i = 0; i < a->rows(); i++) {
      Ciphertext new_b;
      ModSwitchIfNeeded(context, &evaluator, a->value[i], b->value[i], &new_b);

      Ciphertext new_a;
      ModSwitchIfNeeded(context, &evaluator, new_b, a->value[i], &new_a);

      // For add operations the scale needs to be exact, set that here
      new_b.scale() = new_a.scale();

      evaluator.add(new_a, new_b, res.value[i]);
    }

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

    auto data = b.flat<T>().data();
    int rows = b.dim_size(0);
    int cols = b.dim_size(1);

    CipherTensor res(*a);

    seal::Plaintext plain;
    for (int i = 0; i < rows; i++) {
      encoder.encode(
          std::vector<double>(data + (cols * i), data + (cols * (i + 1))),
          kScale, plain);

      seal::Evaluator evaluator(context);

      evaluator.mod_switch_to_inplace(plain, a->value[i].parms_id());

      evaluator.add_plain(a->value[i], plain, res.value[i]);
    }

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

    const PublicKeysVariant* pub_keys = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 2, &pub_keys));

    OP_REQUIRES(ctx, pub_keys->relin_keys.data().size() >= 1,
                InvalidArgument("No relin keys found for seal mul op"));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));

    auto context = SetParams();

    CipherTensor res(*a);

    Evaluator evaluator(context);

    for (int i = 0; i < a->rows(); i++) {
      Ciphertext new_b;
      ModSwitchIfNeeded(context, &evaluator, a->value[i], b->value[i], &new_b);

      Ciphertext new_a;
      ModSwitchIfNeeded(context, &evaluator, new_b, a->value[i], &new_a);

      evaluator.multiply(new_a, new_b, res.value[i]);
      evaluator.relinearize_inplace(res.value[i], pub_keys->relin_keys);
      evaluator.rescale_to_next_inplace(res.value[i]);
    }

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
    seal::Evaluator evaluator(context);
    CipherTensor res(*a);

    auto data = b.flat<T>().data();
    auto rows = b.dim_size(0);
    auto cols = b.dim_size(1);

    seal::Plaintext plain;

    for (int i = 0; i < rows; i++) {
      encoder.encode(
          std::vector<double>(data + (cols * i), data + (cols * (i + 1))),
          kScale, plain);

      evaluator.mod_switch_to_inplace(plain, a->value[i].parms_id());

      evaluator.multiply_plain(a->value[i], plain, res.value[i]);
      evaluator.rescale_to_next_inplace(res.value[i]);
    }

    output->scalar<Variant>()() = std::move(res);
  }
};

// SealMatMulOp and SealMatMulPlainOp expect a transposed b matrix. So
// if you're doing a matmul like [2, 3] x [3, 2] you must first tranpose
// matrix b to [2, 3] and then pass it to the kernel. The reason for this
// has to do with efficency of the algorithm and the layout of data.
class SealMatMulOp : public OpKernel {
 public:
  explicit SealMatMulOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const CipherTensor* a = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 0, &a));

    const CipherTensor* b = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 1, &b));

    OP_REQUIRES(ctx, a->cols() == b->cols(),
                InvalidArgument("Expected a columns to equal b columns saw a ",
                                a->cols(), " and b ", b->cols()));

    const PublicKeysVariant* pub_keys = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 2, &pub_keys));

    OP_REQUIRES(ctx, pub_keys->relin_keys.data().size() >= 1,
                InvalidArgument("No relin keys found for seal matmul op"));

    OP_REQUIRES(ctx, pub_keys->galois_keys.data().size() >= 1,
                InvalidArgument("No galois keys found for seal matmul op"));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));

    auto context = SetParams();

    CipherTensor res(a->rows(), b->rows());
    int rows = a->rows();

    seal::Evaluator evaluator(context);

    matmul(context, &evaluator, *a, *b, &res, pub_keys->relin_keys,
           pub_keys->galois_keys);

    for (int i = 0; i < rows; i++) {
      evaluator.relinearize_inplace(res.value[i], pub_keys->relin_keys);
      evaluator.rescale_to_next_inplace(res.value[i]);
    }

    output->scalar<Variant>()() = std::move(res);
  }
};

template <typename T>
class SealMatMulPlainOp : public OpKernel {
 public:
  explicit SealMatMulPlainOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const CipherTensor* a = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 0, &a));

    const Tensor& b = ctx->input(1);

    OP_REQUIRES(ctx, a->cols() == b.dim_size(1),
                InvalidArgument("Expected a columns to equal b columns saw a ",
                                a->cols(), " and b ", b.dim_size(1)));

    const PublicKeysVariant* pub_keys = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 2, &pub_keys));

    OP_REQUIRES(ctx, pub_keys->galois_keys.data().size() >= 1,
                InvalidArgument("No galois keys found for seal matmul plain op"));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));

    auto context = SetParams();

    seal::CKKSEncoder encoder(context);

    CipherTensor res(a->rows(), b.dim_size(0));

    seal::Evaluator evaluator(context);

    matmul_plain<T>(context, &evaluator, *a, b, &res, pub_keys->galois_keys);

    for (int i = 0; i < a->rows(); i++) {
      evaluator.rescale_to_next_inplace(res.value[i]);
    }

    output->scalar<Variant>()() = std::move(res);
  }
};

// Register the CPU kernels.
#define REGISTER_GENERIC_OPS(T)                                              \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SealEncrypt").Device(DEVICE_CPU).TypeConstraint<T>("dtype"),     \
      SealEncryptOp<T>);                                                     \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SealDecrypt").Device(DEVICE_CPU).TypeConstraint<T>("dtype"),     \
      SealDecryptOp<T>);                                                     \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SealAddPlain").Device(DEVICE_CPU).TypeConstraint<T>("dtype"),    \
      SealAddPlainOp<T>);                                                    \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SealMulPlain").Device(DEVICE_CPU).TypeConstraint<T>("dtype"),    \
      SealMulPlainOp<T>);                                                    \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SealMatMulPlain").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), \
      SealMatMulPlainOp<T>);

REGISTER_GENERIC_OPS(float);
REGISTER_GENERIC_OPS(double);

REGISTER_KERNEL_BUILDER(Name("SealKeyGen").Device(DEVICE_CPU), SealKeyGenOp);
REGISTER_KERNEL_BUILDER(Name("SealAdd").Device(DEVICE_CPU), SealAddOp);
REGISTER_KERNEL_BUILDER(Name("SealMul").Device(DEVICE_CPU), SealMulOp);
REGISTER_KERNEL_BUILDER(Name("SealMatMul").Device(DEVICE_CPU), SealMatMulOp);

}  // namespace tf_seal
