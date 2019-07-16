#include "seal/seal.h"

#include "tf_seal/cc/kernels/seal_tensors.h"

namespace tf_seal {

const char CipherTensor::kTypeName[] = "CipherTensor";

void CipherTensor::Encode(VariantTensorData* data) const {
  // TODO(justin1121) implement this for networking
}

bool CipherTensor::Decode(const VariantTensorData& data) {
  // TODO(justin1121) implement this for networking
  return true;
}

}  // namespace tf_seal
