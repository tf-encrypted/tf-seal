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

RelinKeyVariant::RelinKeyVariant(const RelinKeyVariant& other)
    : keys(other.keys) {}

void RelinKeyVariant::Encode(VariantTensorData* data) const {
  // TODO(justin1121) implement this for networking
}

bool RelinKeyVariant::Decode(const VariantTensorData& data) {
  // TODO(justin1121) implement this for networking
  return true;
}

GaloisKeyVariant::GaloisKeyVariant(const GaloisKeyVariant& other)
    : keys(other.keys) {}

void GaloisKeyVariant::Encode(VariantTensorData* data) const {
  // TODO(justin1121) implement this for networking
}

bool GaloisKeyVariant::Decode(const VariantTensorData& data) {
  // TODO(justin1121) implement this for networking
  return true;
}

const char PublicKeyVariant::kTypeName[] = "SealPublicKeyVariant";
const char SecretKeyVariant::kTypeName[] = "SealSecretKeyVariant";
const char RelinKeyVariant::kTypeName[] = "SealRelinKeyVariant";
const char GaloisKeyVariant::kTypeName[] = "SealGaloisKeyVariant";

}  // namespace tf_seal
