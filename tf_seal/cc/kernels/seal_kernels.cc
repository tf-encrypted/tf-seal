#include <iostream>
#include <vector>
#include <fstream>
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

#include "seal/seal.h"

#include "tf_seal/cc/kernels/key_variants.h"
#include "tf_seal/cc/kernels/seal_context.h"
#include "tf_seal/cc/kernels/seal_helpers.h"
#include "tf_seal/cc/kernels/seal_tensors.h"

namespace tf_seal {

using tensorflow::DEVICE_CPU;
using tensorflow::LookupOrCreateResource;
using tensorflow::MakeTypeIndex;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::ResourceHandle;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TensorShapeUtils;
using tensorflow::Variant;
using tensorflow::core::RefCountPtr;
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

Status LookupOrCreateWrapper(OpKernelContext* ctx,
                             RefCountPtr<Context>* context) {
  auto hndl = ResourceHandle();
  hndl.set_name("seal_context");
  hndl.set_device(ctx->device()->attributes().name());
  hndl.set_hash_code(MakeTypeIndex<Context>().hash_code());

  return tensorflow::LookupOrCreateResource<Context>(ctx, hndl, context,
                                                     [](auto context) {
                                                       *context = new Context;
                                                       return Status();
                                                     });
}

class SealSavePublickeyOp : public OpKernel {
 public:
  explicit SealSavePublickeyOp(OpKernelConstruction* ctx): OpKernel(ctx) {}


  void Compute(OpKernelContext* ctx) override {
      const PublicKeysVariant* key = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 1, &key));
      Tensor* out0;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &out0));
    seal::PublicKey publicKey(key->public_key);
    std::filebuf fb;
    fb.open("public_key", std::ios::out);
    std::ostream pubk(&fb);
    publicKey.save(pubk);
    fb.close();
  }
};


class SealSaveSecretkeyOp : public OpKernel {
 public:
  explicit SealSaveSecretkeyOp(OpKernelConstruction* ctx): OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
     const SecretKeyVariant* secretkey = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 1, &secretkey));
    Tensor* out0;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &out0));
    seal::SecretKey secretKey(secretkey->key);
    std::filebuf fb;
    fb.open("secret_key", std::ios::out);
    std::ostream pubk(&fb);
    secretKey.save(pubk);
    fb.close();
  }
};

class SealSaveCipherTextOp: public OpKernel {
 public:
  explicit SealSaveCipherTextOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
     const SecretKeyVariant* a = nullptr;
     OP_REQUIRES_OK(ctx, GetVariant(ctx, 1, &a));
     Tensor* out0;
     OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &out0));
     // seal::CipherText ciphertext(secretkey->key);
     // std::filebuf fb;
     // fb.open("ciphertext",std::ios::out);
     // std::ostream ciph(&fb);
     // secretKey.save(ciph);
  }
};

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

    RefCountPtr<Context> context;
    OP_REQUIRES_OK(ctx, LookupOrCreateWrapper(ctx, &context));

    seal::KeyGenerator gen(context->context);

    PublicKeysVariant pub_keys;

    if (gen_public) {
      pub_keys.public_key = gen.public_key();
    }

    if (gen_relin) {
      pub_keys.relin_keys = gen.relin_keys();
    }

    if (gen_galois) {
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

    RefCountPtr<Context> context;
    OP_REQUIRES_OK(ctx, LookupOrCreateWrapper(ctx, &context));

    CipherTensor cipher(input.dim_size(0), input.dim_size(1));

    seal::Encryptor encryptor(context->context, key->public_key);
    seal::CKKSEncoder encoder(context->context);

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

    RefCountPtr<Context> context;
    OP_REQUIRES_OK(ctx, LookupOrCreateWrapper(ctx, &context));

    seal::Decryptor decryptor(context->context, key->key);
    seal::CKKSEncoder encoder(context->context);

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

    RefCountPtr<Context> context;
    OP_REQUIRES_OK(ctx, LookupOrCreateWrapper(ctx, &context));

    CipherTensor res(*a);

    for (int i = 0; i < a->rows(); i++) {
      Ciphertext new_b;
      ModSwitchIfNeeded(context->context, &context->evaluator, a->value[i],
                        b->value[i], &new_b);

      Ciphertext new_a;
      ModSwitchIfNeeded(context->context, &context->evaluator, new_b,
                        a->value[i], &new_a);

      // For add operations the scale needs to be exact, set that here
      new_b.scale() = new_a.scale();

      context->evaluator.add(new_a, new_b, res.value[i]);
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

    RefCountPtr<Context> context;
    OP_REQUIRES_OK(ctx, LookupOrCreateWrapper(ctx, &context));

    seal::CKKSEncoder encoder(context->context);

    auto data = b.flat<T>().data();
    int rows = b.dim_size(0);
    int cols = b.dim_size(1);

    CipherTensor res(*a);

    seal::Plaintext plain;
    for (int i = 0; i < rows; i++) {
      encoder.encode(
          std::vector<double>(data + (cols * i), data + (cols * (i + 1))),
          kScale, plain);

      context->evaluator.mod_switch_to_inplace(plain, a->value[i].parms_id());

      context->evaluator.add_plain(a->value[i], plain, res.value[i]);
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

    RefCountPtr<Context> context;
    OP_REQUIRES_OK(ctx, LookupOrCreateWrapper(ctx, &context));

    CipherTensor res(*a);

    auto evaluator = &context->evaluator;

    for (int i = 0; i < a->rows(); i++) {
      Ciphertext new_b;
      ModSwitchIfNeeded(context->context, evaluator, a->value[i], b->value[i],
                        &new_b);

      Ciphertext new_a;
      ModSwitchIfNeeded(context->context, evaluator, new_b, a->value[i],
                        &new_a);

      evaluator->multiply(new_a, new_b, res.value[i]);
      evaluator->relinearize_inplace(res.value[i], pub_keys->relin_keys);
      evaluator->rescale_to_next_inplace(res.value[i]);
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

    RefCountPtr<Context> context;
    OP_REQUIRES_OK(ctx, LookupOrCreateWrapper(ctx, &context));

    seal::CKKSEncoder encoder(context->context);
    auto evaluator = &context->evaluator;

    CipherTensor res(*a);

    auto data = b.flat<T>().data();
    auto rows = b.dim_size(0);
    auto cols = b.dim_size(1);

    seal::Plaintext plain;

    for (int i = 0; i < rows; i++) {
      encoder.encode(
          std::vector<double>(data + (cols * i), data + (cols * (i + 1))),
          kScale, plain);

      evaluator->mod_switch_to_inplace(plain, a->value[i].parms_id());

      evaluator->multiply_plain(a->value[i], plain, res.value[i]);
      evaluator->rescale_to_next_inplace(res.value[i]);
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

    RefCountPtr<Context> context;
    OP_REQUIRES_OK(ctx, LookupOrCreateWrapper(ctx, &context));

    CipherTensor res(a->rows(), b->rows());
    int rows = a->rows();

    auto evaluator = &context->evaluator;

    matmul(context->context, evaluator, *a, *b, &res, pub_keys->relin_keys,
           pub_keys->galois_keys);

    for (int i = 0; i < rows; i++) {
      evaluator->relinearize_inplace(res.value[i], pub_keys->relin_keys);
      evaluator->rescale_to_next_inplace(res.value[i]);
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

    OP_REQUIRES(
        ctx, pub_keys->galois_keys.data().size() >= 1,
        InvalidArgument("No galois keys found for seal matmul plain op"));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));

    auto context = RefCountPtr<Context>();
    OP_REQUIRES_OK(ctx, LookupOrCreateWrapper(ctx, &context));

    seal::CKKSEncoder encoder(context->context);

    CipherTensor res(a->rows(), b.dim_size(0));

    auto evaluator = &context->evaluator;

    matmul_plain<T>(context->context, evaluator, *a, b, &res,
                    pub_keys->galois_keys);

    for (int i = 0; i < a->rows(); i++) {
      evaluator->rescale_to_next_inplace(res.value[i]);
      evaluator->rescale_to_next_inplace(res.value[i]);
    }

    output->scalar<Variant>()() = std::move(res);
  }
};

// Not quite a fully generic PolyEval algorithm. It only supports
// up to four coefficients
// The main issue here is optimizing the computations so that we
// can keep the poly modulus
// degree low. As the poly modulus degree increases the performance
// decreases.
template <typename T>
class SealPolyEvalOp : public OpKernel {
 public:
  explicit SealPolyEvalOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const CipherTensor* x = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 0, &x));

    const Tensor& coeff = ctx->input(1);
    auto data = coeff.flat<T>().data();

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));

    const PublicKeysVariant* pub_keys = nullptr;
    OP_REQUIRES_OK(ctx, GetVariant(ctx, 2, &pub_keys));

    OP_REQUIRES(ctx, pub_keys->relin_keys.data().size() >= 1,
                InvalidArgument("No relin keys found for seal polyeval op"));

    auto context = RefCountPtr<Context>();
    OP_REQUIRES_OK(ctx, LookupOrCreateWrapper(ctx, &context));

    auto evaluator = &context->evaluator;

    seal::CKKSEncoder encoder(context->context);

    // encode the coefficients
    std::vector<Plaintext> coeffs;
    for (int i = 0; i < coeff.NumElements(); i++) {
      Plaintext coeff;

      encoder.encode(data[i], x->value[0].parms_id(), kScale, coeff);
      coeffs.push_back(coeff);
    }

    //
    std::vector<Ciphertext> x1_encrypted(x->rows());
    for(int i = 0; i < x->rows(); i++) {
      evaluator->multiply_plain(x->value[i], coeffs[1], x1_encrypted[i]);

      evaluator->relinearize_inplace(x1_encrypted[i], pub_keys->relin_keys);
      evaluator->rescale_to_next_inplace(x1_encrypted[i]);
      x1_encrypted[i].scale() = kScale;
    }

    std::vector<CipherTensor> xn_encrypteds;
    for (int i = 2; i < coeff.NumElements(); i++) {
      CipherTensor xn_encrypted(x->rows(), x->cols());
      CipherTensor tmp_x_coeff(x->rows(), x->cols());

      std::vector<double> potential_zero(1);
      encoder.decode(coeffs[i], potential_zero);
      if (potential_zero[0] != 0.0) {
        for(int j = 0; j < x->rows(); j++) {
          evaluator->square(x->value[j], xn_encrypted.value[j]);
          evaluator->relinearize_inplace(xn_encrypted.value[j], pub_keys->relin_keys);
          evaluator->rescale_to_next_inplace(xn_encrypted.value[j]);

          if(i == 2) {
            evaluator->multiply_plain_inplace(xn_encrypted.value[j], coeffs[i]);
            evaluator->relinearize_inplace(xn_encrypted.value[j], pub_keys->relin_keys);
            evaluator->rescale_to_next_inplace(xn_encrypted.value[j]);
          } else if(i == 3) {
            evaluator->multiply_plain(x->value[j], coeffs[i], tmp_x_coeff.value[j]);
            evaluator->relinearize_inplace(tmp_x_coeff.value[j], pub_keys->relin_keys);
            evaluator->rescale_to_next_inplace(tmp_x_coeff.value[j]);

            evaluator->multiply_inplace(xn_encrypted.value[j], tmp_x_coeff.value[j]);
            evaluator->relinearize_inplace(xn_encrypted.value[j], pub_keys->relin_keys);
            evaluator->rescale_to_next_inplace(xn_encrypted.value[j]);
          }

          xn_encrypted.value[j].scale() = kScale;
        }

        xn_encrypteds.push_back(xn_encrypted);
      }
    }

    CipherTensor res(x->rows(), x->cols());
    evaluator->mod_switch_to_inplace(coeffs[0], x1_encrypted[0].parms_id());
    for (int i = 0; i < x->rows(); i++) {
      evaluator->add_plain(x1_encrypted[i], coeffs[0], res.value[i]);

      for(auto val: xn_encrypteds) {
        evaluator->mod_switch_to_inplace(res.value[i], val.value[i].parms_id());
        evaluator->add_inplace(res.value[i], val.value[i]);
      }
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
      SealMatMulPlainOp<T>);                                                 \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SealPolyEval").Device(DEVICE_CPU).TypeConstraint<T>("dtype"),    \
      SealPolyEvalOp<T>);

REGISTER_GENERIC_OPS(float);
REGISTER_GENERIC_OPS(double);

REGISTER_KERNEL_BUILDER(Name("SealSaveSecretkey").Device(DEVICE_CPU), SealSaveSecretkeyOp);
REGISTER_KERNEL_BUILDER(Name("SealSavePublickey").Device(DEVICE_CPU), SealSavePublickeyOp);
REGISTER_KERNEL_BUILDER(Name("SealKeyGen").Device(DEVICE_CPU), SealKeyGenOp);
REGISTER_KERNEL_BUILDER(Name("SealAdd").Device(DEVICE_CPU), SealAddOp);
REGISTER_KERNEL_BUILDER(Name("SealMul").Device(DEVICE_CPU), SealMulOp);
REGISTER_KERNEL_BUILDER(Name("SealMatMul").Device(DEVICE_CPU), SealMatMulOp);

}  // namespace tf_seal
