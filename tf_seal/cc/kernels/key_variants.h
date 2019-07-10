#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

#include "seal/seal.h"

using tensorflow::VariantTensorData;

using seal::PublicKey;
using seal::SecretKey;

namespace tf_seal {

class PublicKeyVariant {
 public:
  PublicKeyVariant(PublicKey key) : key(key) {}
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
  SecretKeyVariant(SecretKey key) : key(key) {}
  SecretKeyVariant(const SecretKeyVariant& other);

  static const char kTypeName[];

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  std::string DebugString() const { return "PublicKeyVariant"; }

  SecretKey key;
};

}
