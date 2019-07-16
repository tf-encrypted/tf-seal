import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.framework.errors import InvalidArgumentError

from tf_seal.python.ops.seal_ops import seal_encrypt
from tf_seal.python.ops.seal_ops import seal_decrypt
from tf_seal.python.ops.seal_ops import seal_add
from tf_seal.python.ops.seal_ops import seal_add_plain
from tf_seal.python.ops.seal_ops import seal_mul
from tf_seal.python.ops.seal_ops import seal_mul_plain
from tf_seal.python.ops.seal_ops import seal_key_gen
from tf_seal.python.ops.seal_ops import seal_mat_mul
from tf_seal.python.ops.seal_ops import seal_mat_mul_plain

class SealTest(test.TestCase):
  """SealTest test"""

  def test_encrypt_decrypt(self):
    with tf.Session() as sess:
      inp = np.array([[44.44, 55.55], [66.66, 77.77]], np.float32)
      pub_key, sec_key = seal_key_gen()
      variant = seal_encrypt(inp, pub_key)
      ans = seal_decrypt(variant, sec_key, tf.float32)

      np.testing.assert_equal(sess.run(ans), inp)

  def test_add(self):
    with tf.Session() as sess:
      a = np.array([[44.44]], np.float64)
      b = np.array([[44.44]], np.float64)

      ans = a + b

      pub_key, sec_key = seal_key_gen()

      a_var = seal_encrypt(a, pub_key)
      b_var = seal_encrypt(b, pub_key)

      c_var = seal_add(a_var, b_var)

      c = seal_decrypt(c_var, sec_key, tf.float64)

      np.testing.assert_almost_equal(sess.run(c), ans, 0.001)

  def test_add_plain(self):
    with tf.Session() as sess:
      a = np.array([[44.44, 55.55], [66.66, 77.77]], np.float32)
      b = np.array([[44.44, 55.55], [66.66, 77.77]], np.float32)

      ans = a + b

      pub_key, sec_key = seal_key_gen()

      a_var = seal_encrypt(a, pub_key)

      c_var = seal_add_plain(a_var, b)

      c = seal_decrypt(c_var, sec_key, tf.float32)

      np.testing.assert_almost_equal(sess.run(c), ans, 0.001)

  def test_mul(self):
    with tf.Session() as sess:
      a = np.array([[44.44, 55.55], [66.66, 77.77]], np.float64)
      b = np.array([[44.44, 55.55], [66.66, 77.77]], np.float64)
      c = np.array([[44.44, 55.55], [66.66, 77.77]], np.float64)

      ans = a * b * c

      pub_key, sec_key = seal_key_gen(gen_relin=True)

      a_var = seal_encrypt(a, pub_key)
      b_var = seal_encrypt(b, pub_key)
      c_var = seal_encrypt(c, pub_key)

      tmp = seal_mul(a_var, b_var, pub_key)
      d_var = seal_mul(tmp, c_var, pub_key)

      d = seal_decrypt(d_var, sec_key, tf.float64)

      np.testing.assert_almost_equal(sess.run(d), ans, 0.001)

  def test_mul_then_add(self):
    with tf.Session() as sess:
      a = np.array([[44.44, 55.55], [66.66, 77.77]], np.float64)
      b = np.array([[44.44, 55.55], [66.66, 77.77]], np.float64)
      c = np.array([[44.44, 55.55], [66.66, 77.77]], np.float64)

      ans = a * b + c

      pub_key, sec_key = seal_key_gen(gen_relin=True)

      a_var = seal_encrypt(a, pub_key)
      b_var = seal_encrypt(b, pub_key)
      c_var = seal_encrypt(c, pub_key)

      tmp = seal_mul(a_var, b_var, pub_key)
      d_var = seal_add(tmp, c_var)

      d = seal_decrypt(d_var, sec_key, tf.float64)

      np.testing.assert_almost_equal(sess.run(d), ans, 0.001)

  def test_mul_plain(self):
    with tf.Session() as sess:
      a = np.array([[44.44, 55.55], [66.66, 77.77]], np.float32)
      b = np.array([[44.44, 55.55], [66.66, 77.77]], np.float32)

      ans = a * b

      pub_key, sec_key = seal_key_gen()

      a_var = seal_encrypt(a, pub_key)

      c_var = seal_mul_plain(a_var, b)

      c = seal_decrypt(c_var, sec_key, tf.float32)

      np.testing.assert_almost_equal(sess.run(c), ans, 0.001)

  def test_matmul(self):
    with tf.Session() as sess:
      a = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
      b = np.array([[1, 2], [3, 4], [5, 6]], np.float32)

      ans = a.dot(b)

      pub_key, sec_key = seal_key_gen(gen_relin=True, gen_galois=True)

      a_var = seal_encrypt(a, pub_key)
      b_var = seal_encrypt(b.transpose(), pub_key)

      c_var = seal_mat_mul(a_var, b_var, pub_key)

      c = seal_decrypt(c_var, sec_key, tf.float32)

      np.testing.assert_almost_equal(sess.run(c), ans, 0.001)

  def test_matmul_plain(self):
    with tf.Session() as sess:
      a = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
      b = np.array([[1, 2], [3, 4], [5, 6]], np.float32)

      ans = a.dot(b)

      pub_key, sec_key = seal_key_gen(gen_galois=True)

      a_var = seal_encrypt(a, pub_key)

      c_var = seal_mat_mul_plain(a_var, b.transpose(), pub_key)

      c = seal_decrypt(c_var, sec_key, tf.float32)

      np.testing.assert_almost_equal(sess.run(c), ans, 0.001)

  def test_matmul_then_add(self):
    with tf.Session() as sess:
      a = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
      b = np.array([[1, 2], [3, 4], [5, 6]], np.float32)
      c = np.array([[1, 2], [4, 5]], np.float32)

      ans = a.dot(b) + c

      pub_key, sec_key = seal_key_gen(gen_relin=True, gen_galois=True)

      a_var = seal_encrypt(a, pub_key)
      b_var = seal_encrypt(b.transpose(), pub_key)
      c_var = seal_encrypt(c, pub_key)

      tmp = seal_mat_mul(a_var, b_var, pub_key)
      d_var = seal_add(tmp, c_var)

      d = seal_decrypt(d_var, sec_key, tf.float32)

      np.testing.assert_almost_equal(sess.run(d), ans, 0.001)

  def test_incompatible_shape(self):
    with tf.Session() as sess:
      a = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
      b = np.array([[1, 2], [3, 4], [5, 6]], np.float32)

      pub_key, _ = seal_key_gen()

      a_var = seal_encrypt(a, pub_key)
      b_var = seal_encrypt(b, pub_key)

      tmp = seal_mat_mul(a_var, b_var, pub_key)

      try:
        sess.run(tmp)
      except InvalidArgumentError:
        pass


if __name__ == '__main__':
  test.main()
