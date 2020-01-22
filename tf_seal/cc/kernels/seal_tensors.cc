#include "seal/seal.h"

#include "tf_seal/cc/kernels/seal_tensors.h"

namespace tf_seal {

const char CipherTensor::kTypeName[] = "CipherTensor";

void CipherTensor::Encode(tensorflow::VariantTensorData* data) const {
  // TODO(justin1121) implement this for networking
}

bool CipherTensor::Decode(const tensorflow::VariantTensorData& data) {
  // TODO(justin1121) implement this for networking
  return true;
}

void CipherTensor::Encode(proto::EncryptedTensor* buf) const {
  buf->set_rows(_rows);
  buf->set_cols(_cols);

  for (auto ciphertext : value) {
    std::ostringstream stream;
    ciphertext.save(stream);
    buf->add_seal_ciphertext(stream.str());
  }
}

bool CipherTensor::Decode(
    const std::shared_ptr<seal::SEALContext> seal_context,
    const proto::EncryptedTensor& buf)
{
  _rows = buf.rows();
  _cols = buf.cols();

  value.clear();
  for (auto serialized_ciphertext : buf.seal_ciphertext()) {
    std::istringstream stream(serialized_ciphertext);
    seal::Ciphertext ciphertext;
    ciphertext.load(seal_context, stream);
    value.push_back(ciphertext);
  }

  return true;
}

}  // namespace tf_seal
