#include "seal/seal.h"

#include "tf_seal/cc/kernels/key_variants.h"

namespace tf_seal {

PublicKeyVariant::PublicKeyVariant(const PublicKeyVariant& other)
    : key(other.key) {}

void PublicKeyVariant::Encode(VariantTensorData* data) const {
  // TODO(justin1121) implement this for networking
}

bool PublicKeyVariant::Decode(const VariantTensorData& data) {
  // TODO(justin1121) implement this for networking
  return true;
}

SecretKeyVariant::SecretKeyVariant(const SecretKeyVariant& other)
    : key(other.key) {}

void SecretKeyVariant::Encode(VariantTensorData* data) const {
  // TODO(justin1121) implement this for networking
}

bool SecretKeyVariant::Decode(const VariantTensorData& data) {
  // TODO(justin1121) implement this for networking
  return true;
}

const char PublicKeyVariant::kTypeName[] = "PublicKeyVariant";
const char SecretKeyVariant::kTypeName[] = "SecretKeyVariant";

}