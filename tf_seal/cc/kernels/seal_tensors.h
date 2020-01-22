#ifndef TF_SEAL_CC_KERNELS_SEAL_TENSORS_H_
#define TF_SEAL_CC_KERNELS_SEAL_TENSORS_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

#include "seal/seal.h"


namespace tf_seal {

class CipherTensor {
 public:
  CipherTensor(int rows, int cols) : value(rows), _rows(rows), _cols(cols) {}

  CipherTensor(const CipherTensor& other)
      : value(other.value), _rows(other._rows), _cols(other._cols) {}

  static const char kTypeName[];

  std::string TypeName() const { return kTypeName; }

  void Encode(tensorflow::VariantTensorData* data) const;

  bool Decode(const tensorflow::VariantTensorData& data);


  std::string DebugString() const { return "CipherTensor"; }

  int rows() const { return _rows; }
  int cols() const { return _cols; }

  std::vector<seal::Ciphertext> value;

 private:
  int _rows;
  int _cols;
};

}  // namespace tf_seal

#endif  // TF_SEAL_CC_KERNELS_SEAL_TENSORS_H_
