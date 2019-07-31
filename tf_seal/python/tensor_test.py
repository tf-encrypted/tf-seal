import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import test

from tf_seal.python.ops.seal_ops import seal_encrypt, seal_key_gen
from tf_seal.python.tensor import convert_to_tensor


class EvaluationTest(test.TestCase):

  def test_session_run(self):
    public_keys, secret_key = seal_key_gen()
    x_raw = np.array([[1234, 1234]]).astype(np.float64)
    x = convert_to_tensor(x_raw, secret_key, public_keys)

    with tf.Session() as sess:
      res = sess.run(x)
      np.testing.assert_array_almost_equal(res, x_raw.astype(np.float64), 0.1)

  def test_eval(self):
    public_keys, secret_key = seal_key_gen()
    x_raw = np.array([[1234, 1234]]).astype(np.float64)
    x = convert_to_tensor(x_raw, secret_key, public_keys)

    with tf.Session() as sess:
      res = x.eval(session=sess)
      np.testing.assert_array_almost_equal(res, x_raw.astype(np.float64), 0.1)


class ArithmeticTest(test.TestCase):
  def _core_matmul_test(self, plain=False):
    public_keys, secret_key = seal_key_gen(gen_relin=True, gen_galois=True)

    x_raw = np.array([[1234, 1234], [1234, 1234]]).astype(np.float64)
    y_raw = np.array([[1234, 1234], [1234, 1234]]).astype(np.float64)
    z_raw = x_raw.dot(y_raw)

    x = convert_to_tensor(x_raw, secret_key, public_keys)

    if not plain:
        y = convert_to_tensor(y_raw, secret_key, public_keys)
    else:
        y = y_raw

    z = x.matmul(y)

    with tf.Session() as sess:
      res = sess.run(z)
      np.testing.assert_array_almost_equal(res, z_raw.astype(np.float64), 0.1)

  def _core_test(self, op, plain=False):
    public_keys, secret_key = seal_key_gen(gen_relin=True)

    x_raw = np.array([[1234, 1234]]).astype(np.float64)
    y_raw = np.array([[1234, 1234]]).astype(np.float64)
    z_raw = op(x_raw, y_raw)

    x = convert_to_tensor(x_raw, secret_key, public_keys)

    if not plain:
        y = convert_to_tensor(y_raw, secret_key, public_keys)
    else:
        y = y_raw

    z = op(x, y)

    with tf.Session() as sess:
      res = sess.run(z)
      np.testing.assert_array_almost_equal(res, z_raw.astype(np.float64), 0.1)

  def test_add(self):
    self._core_test(lambda x, y: x + y)

  def test_add_plain(self):
    self._core_test(lambda x, y: x + y, plain=True)

  def test_mul(self):
    self._core_test(lambda x, y: x * y)

  def test_mul_plain(self):
    self._core_test(lambda x, y: x * y, plain=True)

  def test_matmul(self):
    self._core_matmul_test()

  def test_matmul_plain(self):
    self._core_matmul_test(plain=True)


class ConvertTest(test.TestCase):

  def _core_test(self, in_np, out_np, convert_to_tf_tensor):
    public_keys, secret_key = seal_key_gen()

    if convert_to_tf_tensor:
      in_tf = tf.convert_to_tensor(in_np)
      x = convert_to_tensor(in_tf, secret_key, public_keys)
    else:
      x = convert_to_tensor(in_np, secret_key, public_keys)

    with tf.Session() as sess:
      res = sess.run(x)
      np.testing.assert_array_almost_equal(res, out_np, 0.1)

  def test_constant_float32(self):
    x = np.array([[1,2,3,4]]).astype(np.float32)
    self._core_test(
      in_np=x,
      out_np=x,
      convert_to_tf_tensor=False,
    )
    self._core_test(
      in_np=x,
      out_np=x,
      convert_to_tf_tensor=True,
    )

  def test_constant_float64(self):
    x = np.array([[1,2,3,4]]).astype(np.float64)
    self._core_test(
      in_np=x,
      out_np=x.astype(np.float64),
      convert_to_tf_tensor=False,
    )
    self._core_test(
      in_np=x,
      out_np=x.astype(np.float64),
      convert_to_tf_tensor=True,
    )

  def test_constant_numpy_object(self):
    x = np.array([[1234]]).astype(np.float64)
    self._core_test(
      in_np=x,
      out_np=x.astype(np.float64),
      convert_to_tf_tensor=False,
    )

#   def test_is_tensor(self):
#     x = convert_to_tensor(np.array([[10, 20]]))
#     #assert tf.is_tensor(x)  # for TensorFlow >=1.14
#     assert tf.contrib.framework.is_tensor(x)

#   def test_register_tensor_conversion_function(self):
#     x = convert_to_tensor(np.array([[10, 20]]))
#     y = tf.convert_to_tensor(np.array([[30, 40]]))
#     z = x + y
#     with tf.Session() as sess:
#       res = sess.run(z)
#       np.testing.assert_array_equal(res, np.array([["40", "60"]]))

#   def test_convert_to_tensor(self):
#     x = convert_to_tensor(np.array([[10, 20]]))
#     y = tf.convert_to_tensor(x)
#     assert y.dtype is tf.string


# class IntegrationTest(test.TestCase):

#   def test_register_symbolic(self):
#     x = convert_to_tensor(np.array(10))
#     assert tf_utils.is_symbolic_tensor(x)

#   # def test_use_in_model(self):
#   #   x = convert_to_tensor(np.array(10))
#   #   model = tf.keras.models.Sequential([
#   #     tf.keras.layers.Dense(10)
#   #   ])
#   #   model(x)


if __name__ == '__main__':
  test.main()