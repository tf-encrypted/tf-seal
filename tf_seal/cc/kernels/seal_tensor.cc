#include "seal/seal.h"

#include "tf_seal/cc/kernels/seal_tensor.h"

namespace tf_seal {
SealTensor::SealTensor(const SealTensor& other)
    : value(other.value),
      sec_key(other.sec_key),
      pub_key(other.pub_key),
      _rows(other._rows),
      _cols(other._cols) {}

void SealTensor::Encode(VariantTensorData* data) const {
  // TODO(justin1121) implement this for networking
}

bool SealTensor::Decode(const VariantTensorData& data) {
  // TODO(justtin1121) implement this for networking
  return true;
}

const char SealTensor::kTypeName[] = "SealTensor";
}  // namespace tf_seal
