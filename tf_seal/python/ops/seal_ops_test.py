import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.framework.errors import InvalidArgumentError

from tf_seal.python.ops.seal_ops import seal_key_gen
from tf_seal.python.ops.seal_ops import seal_save_publickey

from tf_seal.python.ops.seal_ops import seal_encrypt
from tf_seal.python.ops.seal_ops import seal_decrypt
from tf_seal.python.ops.seal_ops import seal_add
from tf_seal.python.ops.seal_ops import seal_add_plain
from tf_seal.python.ops.seal_ops import seal_mul
from tf_seal.python.ops.seal_ops import seal_mul_plain
from tf_seal.python.ops.seal_ops import seal_mat_mul
from tf_seal.python.ops.seal_ops import seal_mat_mul_plain
from tf_seal.python.ops.seal_ops import seal_poly_eval

class SealTest(test.TestCase):
  """SealTest test"""

  def test_save_pubkey(self):
    _, tmp_filename = tempfile.mkstemp()

    with tf.Session() as sess:
      pubkey, _ = seal_key_gen()
      save_op = seal_save_publickey(tmp_filename, pubkey)
      sess.run(save_op)

    assert os.path.isfile(tmp_filename), \
        "Did not find expected file: '{}'".format(
            tmp_filename)
    assert os.path.getsize(tmp_filename) > 100, \
        "File smaller than expected: '{}', size: {}".format(
            tmp_filename, os.path.getsize(tmp_filename))

    os.remove(tmp_filename)

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

      np.testing.assert_almost_equal(sess.run(c), ans, 0.1)

  def test_add_plain(self):
    with tf.Session() as sess:
      a = np.array([[44.44, 55.55], [66.66, 77.77]], np.float32)
      b = np.array([[44.44, 55.55], [66.66, 77.77]], np.float32)

      ans = a + b

      pub_key, sec_key = seal_key_gen()

      a_var = seal_encrypt(a, pub_key)

      c_var = seal_add_plain(a_var, b)

      c = seal_decrypt(c_var, sec_key, tf.float32)

      np.testing.assert_almost_equal(sess.run(c), ans, 0.1)

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

      np.testing.assert_almost_equal(sess.run(d), ans, 0.1)

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

      np.testing.assert_almost_equal(sess.run(d), ans, 0.1)

  def test_mul_plain(self):
    with tf.Session() as sess:
      a = np.array([[44.44, 55.55], [66.66, 77.77]], np.float32)
      b = np.array([[44.44, 55.55], [66.66, 77.77]], np.float32)

      ans = a * b

      pub_key, sec_key = seal_key_gen()

      a_var = seal_encrypt(a, pub_key)

      c_var = seal_mul_plain(a_var, b)

      c = seal_decrypt(c_var, sec_key, tf.float32)

      np.testing.assert_almost_equal(sess.run(c), ans, 0.1)

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

      np.testing.assert_almost_equal(sess.run(c), ans, 0.1)

  def test_matmul_plain(self):
    with tf.Session() as sess:
      a = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
      b = np.array([[1, 2], [3, 4], [5, 6]], np.float32)

      ans = a.dot(b)

      pub_key, sec_key = seal_key_gen(gen_galois=True)

      a_var = seal_encrypt(a, pub_key)

      c_var = seal_mat_mul_plain(a_var, b.transpose(), pub_key)

      c = seal_decrypt(c_var, sec_key, tf.float32)

      np.testing.assert_almost_equal(sess.run(c), ans, 0.1)

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

      np.testing.assert_almost_equal(sess.run(d), ans, 0.1)

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

  def test_poly_eval(self):
    with tf.Session() as sess:
      x = np.array([[1, 2, 3], [4, 5, 5]], np.float32)
      coeffs = np.array([0.5, 0.197, 0.0, -0.004])

      ans = tf.sigmoid(x)

      pub_key, sec_key = seal_key_gen(gen_relin=True, gen_galois=True)

      x_var = seal_encrypt(x, pub_key)

      c_var = seal_poly_eval(x_var, coeffs, pub_key)

      c = seal_decrypt(c_var, sec_key, tf.float32)

      np.testing.assert_almost_equal(sess.run(c), sess.run(ans), 0.1)

  def test_matmul_poly_eval(self):
    with tf.Session() as sess:
      x = np.array([[0.02, 0.04, 0.05], [0.06, 1, 0.0005]], np.float32)
      y = np.array([[0.02, 0.04], [0.05, 0.06], [1, 0.0005]], np.float32)
      coeffs = np.array([0.5, 0.197, 0.0, -0.004])

      m = tf.matmul(x, y)
      ans = tf.sigmoid(m)

      pub_key, sec_key = seal_key_gen(gen_relin=True, gen_galois=True)

      x_var = seal_encrypt(x, pub_key)

      z_var = seal_mat_mul_plain(x_var, y.transpose(), pub_key)

      c_var = seal_poly_eval(z_var, coeffs, pub_key)

      c = seal_decrypt(c_var, sec_key, tf.float32)

      np.testing.assert_almost_equal(sess.run(c).transpose(), sess.run(ans), 0.1)


if __name__ == '__main__':
  test.main()
