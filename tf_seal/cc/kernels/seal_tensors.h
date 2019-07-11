#ifndef TF_SEAL_CC_KERNELS_SEAL_TENSORS_H_
#define TF_SEAL_CC_KERNELS_SEAL_TENSORS_H_

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

using seal::Ciphertext;

class SealTensor {
 public:
  SealTensor(int rows, int cols) : _rows(rows), _cols(cols) {}
  SealTensor(const SealTensor& other);

  // needs a virtual method for the class to be polymorphic
  virtual ~SealTensor() = default;

  static const char kTypeName[];

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  std::string DebugString() const { return "SealTensor"; }

  int rows() const { return _rows; }

  int cols() const { return _cols; }

 private:
  int _rows;
  int _cols;
};

class CipherTensor : public SealTensor {
 public:
  using SealTensor::SealTensor;

  CipherTensor(const CipherTensor& other);

  static const char kTypeName[];

  std::string DebugString() const { return "CipherTensor"; }

  Ciphertext value;
};

}  // namespace tf_seal

#endif  // TF_SEAL_CC_KERNELS_SEAL_TENSORS_H_
