#include "seal/seal.h"

#include "tf_seal/cc/kernels/key_variants.h"

namespace tf_seal {

void PublicKeysVariant::Encode(VariantTensorData* data) const {
  // TODO(justin1121) implement this for networking
}

bool PublicKeysVariant::Decode(const VariantTensorData& data) {
  // TODO(justin1121) implement this for networking
  return true;
}

void SecretKeyVariant::Encode(VariantTensorData* data) const {
  // TODO(justin1121) implement this for networking
}

bool SecretKeyVariant::Decode(const VariantTensorData& data) {
  // TODO(justin1121) implement this for networking
  return true;
}

const char PublicKeysVariant::kTypeName[] = "SealPublicKeysVariant";
const char SecretKeyVariant::kTypeName[] = "SealSecretKeyVariant";

}  // namespace tf_seal
