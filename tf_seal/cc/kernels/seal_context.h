#ifndef TF_SEAL_CC_KERNELS_SEAL_CONTEXT_H_
#define TF_SEAL_CC_KERNELS_SEAL_CONTEXT_H_

#include <memory>
#include <string>

#include "tensorflow/core/framework/resource_mgr.h"

#include "seal/seal.h"

#include "tf_seal/cc/kernels/seal_helpers.h"

namespace tf_seal {

std::shared_ptr<seal::SEALContext> SetParams() {
  seal::EncryptionParameters parms(seal::scheme_type::CKKS);

  parms.set_poly_modulus_degree(kPolyModulusDegree);
  parms.set_coeff_modulus(
      seal::CoeffModulus::Create(kPolyModulusDegree, {60, 40, 40, 60}));

  return seal::SEALContext::Create(parms);
}

struct Context : public tensorflow::ResourceBase {
  Context() : context(SetParams()), evaluator(context) {}

  // Returns a debug string for *this.
  virtual std::string DebugString() const { return "SEAL Context"; }

  // Returns memory used by this resource.
  // TODO(justin1121): consider estimating this
  virtual tensorflow::int64 MemoryUsed() const { return 0; }

  std::shared_ptr<seal::SEALContext> context;
  Evaluator evaluator;
};
}  // namespace tf_seal

#endif  // TF_SEAL_CC_KERNELS_SEAL_CONTEXT_H_
