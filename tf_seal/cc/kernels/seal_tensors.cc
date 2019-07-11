#include "seal/seal.h"

#include "tf_seal/cc/kernels/seal_tensors.h"

namespace tf_seal {
SealTensor::SealTensor(const SealTensor& other)
    : _rows(other._rows), _cols(other._cols) {}

void SealTensor::Encode(VariantTensorData* data) const {
  // TODO(justin1121) implement this for networking
}

bool SealTensor::Decode(const VariantTensorData& data) {
  // TODO(justin1121) implement this for networking
  return true;
}

const char SealTensor::kTypeName[] = "SealTensor";

const char CipherTensor::kTypeName[] = "CipherTensor";

CipherTensor::CipherTensor(const CipherTensor& other)
    : SealTensor(other), value(other.value) {}

}  // namespace tf_seal
