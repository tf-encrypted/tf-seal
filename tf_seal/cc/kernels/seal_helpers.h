#ifndef TF_SEAL_CC_KERNELS_SEAL_HELPERS_H_
#define TF_SEAL_CC_KERNELS_SEAL_HELPERS_H_
#include <memory>
#include <vector>

#include "seal/seal.h"

#include "tf_seal/cc/kernels/seal_tensors.h"

namespace tf_seal {

using tensorflow::Tensor;

using seal::Ciphertext;
using seal::CKKSEncoder;
using seal::Evaluator;
using seal::GaloisKeys;
using seal::Plaintext;
using seal::RelinKeys;
using seal::SEALContext;

const double kScale = pow(2.0, 40);
const size_t kPolyModulusDegreePower = 13;
const size_t kPolyModulusDegree = pow(2, kPolyModulusDegreePower);

// Algorithm 4 (FHE.sumslots)
// from https://eprint.iacr.org/2018/462
void rotate_sum(Evaluator* evaluator, Ciphertext* cipher,
                const GaloisKeys& keys) {
  Ciphertext rotated;
  for (int i = 0; i < kPolyModulusDegreePower - 1; i++) {
    evaluator->rotate_vector(*cipher, pow(2, i), keys, rotated);
    evaluator->add_inplace(*cipher, rotated);
  }
}

void zero_all_but_first(std::shared_ptr<SEALContext> context,
                        Evaluator* evaluator, Ciphertext* cipher,
                        double scale) {
  CKKSEncoder encoder(context);

  std::vector<double> one_and_zeros(1);
  one_and_zeros[0] = 1.0;

  Plaintext plain;

  // is there a way to keep the encoding smaller than 2 ^ 40 for ones and zeros?
  encoder.encode(one_and_zeros, pow(2, 40), plain);

  evaluator->multiply_plain_inplace(*cipher, plain);
}

// Matmul expects a column major order matrix for tensor b.
void matmul(std::shared_ptr<SEALContext> context, Evaluator* evaluator,
            const CipherTensor& a, const CipherTensor& b, CipherTensor* c,
            const RelinKeys& relin_keys, const GaloisKeys& galois_keys) {
  CKKSEncoder encoder(context);

  int rows_a = a.rows(), cols_b = b.rows();

  for (int i = 0; i < rows_a; i++) {
    std::vector<Ciphertext> tmp(cols_b);

    Plaintext zeros;
    for (int j = 0; j < cols_b; j++) {
      // first multiple row i and column j
      evaluator->multiply(a.value[i], b.value[j], tmp[j]);

      // relinearize to shrink the cipher size back to 2
      evaluator->relinearize_inplace(tmp[j], relin_keys);

      // calculate the sum, the sum will end up in every slot of the cipher
      rotate_sum(evaluator, &(tmp[j]), galois_keys);

      // we only want the sum in one slot so multiple by ones and zeros
      // so that the sum is in the first slot
      zero_all_but_first(context, evaluator, &(tmp[j]), kScale);

      // rescale once
      evaluator->rescale_to_next_inplace(tmp[j]);

      // next we need to fill in the final row so we first create temporary
      // cipher with all zeros and set the first element to the first element in
      // tmp
      if (j == 0) {
        encoder.encode(std::vector<double>(0), tmp[j].parms_id(),
                       tmp[j].scale(), zeros);

        evaluator->add_plain(tmp[j], zeros, c->value[i]);
      } else {
        // for every other element we rotate tmp to the right by j and then add
        // it to the output row
        evaluator->rotate_vector_inplace(tmp[j], -j, galois_keys);

        evaluator->add_inplace(c->value[i], tmp[j]);
      }
    }
  }
}

// matmul plain is very similar to the above matmul, see it for more details
template <typename T>
void matmul_plain(std::shared_ptr<SEALContext> context, Evaluator* evaluator,
                  const CipherTensor& a, const Tensor& b, CipherTensor* c,
                  const GaloisKeys& galois_keys) {
  CKKSEncoder encoder(context);

  int rows_a = a.rows(), cols_b = b.dim_size(0), rows_b = b.dim_size(1);

  auto data = b.flat<T>().data();

  for (int i = 0; i < rows_a; i++) {
    std::vector<Ciphertext> tmp(cols_b);

    Plaintext zeros;
    for (int j = 0; j < cols_b; j++) {
      Plaintext b_plain;
      encoder.encode(
          std::vector<double>(data + (rows_b * j), data + (rows_b * (j + 1))),
          kScale, b_plain);
      evaluator->multiply_plain(a.value[i], b_plain, tmp[j]);

      rotate_sum(evaluator, &(tmp[j]), galois_keys);

      zero_all_but_first(context, evaluator, &(tmp[j]), kScale);

      evaluator->rescale_to_next_inplace(tmp[j]);

      if (j == 0) {
        encoder.encode(std::vector<double>(0), tmp[j].parms_id(),
                       tmp[j].scale(), zeros);

        evaluator->add_plain(tmp[j], zeros, c->value[i]);
      } else {
        evaluator->rotate_vector_inplace(tmp[j], -j, galois_keys);

        evaluator->add_inplace(c->value[i], tmp[j]);
      }
    }
  }
}

}  // namespace tf_seal

#endif  // TF_SEAL_CC_KERNELS_SEAL_HELPERS_H_
