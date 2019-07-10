#ifndef TF_SEAL_CC_KERNELS_SEAL_TENSOR_H_
#define TF_SEAL_CC_KERNELS_SEAL_TENSOR_H_

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

using seal::Ciphertext;
using seal::PublicKey;
using seal::SecretKey;

namespace tf_seal {

struct SealTensor {
  SealTensor(int rows, int cols) : _rows(rows), _cols(cols) {}
  SealTensor(const SealTensor& other);

  static const char kTypeName[];
  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  std::string DebugString() const { return "SealTensor"; }

  int rows() const { return _rows; }

  int cols() const { return _cols; }

  Ciphertext value;

  // TODO(justin1121) Keys should probably be in their own variant
  SecretKey sec_key;
  PublicKey pub_key;

 private:
  int _rows;
  int _cols;
};

}  // namespace tf_seal

#endif  // TF_SEAL_CC_KERNELS_SEAL_TENSOR_H_
