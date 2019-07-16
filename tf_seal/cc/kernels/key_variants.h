#ifndef TF_SEAL_CC_KERNELS_KEY_VARIANTS_H_
#define TF_SEAL_CC_KERNELS_KEY_VARIANTS_H_

#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

#include "seal/seal.h"

namespace tf_seal {

using tensorflow::VariantTensorData;

using seal::GaloisKeys;
using seal::PublicKey;
using seal::RelinKeys;
using seal::SecretKey;

class PublicKeyVariant {
 public:
  explicit PublicKeyVariant(PublicKey key) : key(key) {}
  PublicKeyVariant(const PublicKeyVariant& other);

  static const char kTypeName[];

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  std::string DebugString() const { return "PublicKeyVariant"; }

  PublicKey key;
};

class SecretKeyVariant {
 public:
  explicit SecretKeyVariant(SecretKey key) : key(key) {}
  SecretKeyVariant(const SecretKeyVariant& other);

  static const char kTypeName[];

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  std::string DebugString() const { return "SecretKeyVariant"; }

  SecretKey key;
};

class RelinKeyVariant {
 public:
  explicit RelinKeyVariant(RelinKeys keys) : keys(keys) {}
  RelinKeyVariant(const RelinKeyVariant& other);

  static const char kTypeName[];

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  std::string DebugString() const { return "RelinKeyVariant"; }

  RelinKeys keys;
};

class GaloisKeyVariant {
 public:
  explicit GaloisKeyVariant(GaloisKeys keys) : keys(keys) {}
  GaloisKeyVariant(const GaloisKeyVariant& other);

  static const char kTypeName[];

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  std::string DebugString() const { return "GaloisKeyVariant"; }

  GaloisKeys keys;
};

}  // namespace tf_seal

#endif  // TF_SEAL_CC_KERNELS_KEY_VARIANTS_H_
