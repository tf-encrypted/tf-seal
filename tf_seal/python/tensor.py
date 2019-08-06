import numpy as np
import tensorflow as tf

import tf_seal.python.ops.seal_ops as ops

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import ops as tf_ops


class Tensor(object):

  def __init__(self, value, secret_key, public_keys):
    assert isinstance(value, tf.Tensor), type(value)
    assert value.dtype is tf.variant, value.dtype
    self._raw = value
    self._public_keys = public_keys
    self._secret_key = secret_key

  @property
  def shape(self):
    return self._raw.shape

  @property
  def name(self):
    return self._raw.name

  @property
  def dtype(self):
    return tf.int32
    # return tf.string

  def eval(self, session=None, dtype=None):
    tf_tensor = convert_from_tensor(self, dtype=dtype)
    evaluated = tf_tensor.eval(session=session)

    return evaluated

  def __add__(self, other):
    if isinstance(other, Tensor):
      res = ops.seal_add(self._raw, other._raw)
    else:
      res = ops.seal_add_plain(self._raw, other)

    return Tensor(res, self._secret_key, self._public_keys)

#   def __sub__(self, other):
#     other = convert_to_tensor(other)
#     res = ops.big_sub(self._raw, other._raw)
#     return Tensor(res)

  def __mul__(self, other):
    if isinstance(other, Tensor):
      res = ops.seal_mul(self._raw, other._raw, self._public_keys)
    else:
      res = ops.seal_mul_plain(self._raw, other)
    return Tensor(res, self._secret_key, self._public_keys)

  def matmul(self, other):
    if isinstance(other, Tensor):
      res = ops.seal_mat_mul(self._raw, other._raw, self._public_keys)
    else:
      res = ops.seal_mat_mul_plain(self._raw, other, self._public_keys)
    return Tensor(res, self._secret_key, self._public_keys)


def _fetch_function(seal_tensor):
  unwrapped = [convert_from_tensor(seal_tensor, dtype=tf.float64)]
  rewrapper = lambda components_fetched: components_fetched[0].astype(np.float64)
  return unwrapped, rewrapper

def _feed_function(seal_tensor, feed_value):
  return [(seal_tensor._raw, feed_value)]

def _feed_function_for_partial_run(seal_tensor):
  return [seal_tensor._raw]

# this allows tf_seal.Tensor to be passed directly to tf.Session.run,
# unwrapping and converting the result as needed
tf_session.register_session_run_conversion_functions(
    tensor_type=Tensor,
    fetch_function=_fetch_function,
    feed_function=_feed_function,
    feed_function_for_partial_run=_feed_function_for_partial_run,
)


def _tensor_conversion_function(tensor, dtype=None, name=None, as_ref=False):
  assert name is None, "Not implemented, name='{}'".format(name)
  assert not as_ref, "Not implemented, as_ref={}".format(as_ref)
  assert dtype in [tf.float32, tf.float64, None], dtype
  return convert_from_tensor(tensor, dtype=dtype)

# TODO(Morten)
# this allows implicit convertion of tf_seal.Tensor to tf.Tensor,
# but since the output dtype is determined by the outer context
# we essentially have to export with the implied risk of data loss
tf_ops.register_tensor_conversion_function(Tensor, _tensor_conversion_function)


# this allows Tensor to pass the tf.is_tensor test
tf_ops.register_dense_tensor_like_type(Tensor)


# this allows tf_big.Tensor to be plumbed through Keras layers
# but seems only truly useful when used in conjunction with
# `register_tensor_conversion_function`
tf_utils.register_symbolic_tensor_type(Tensor)


def constant(tensor, secret_key, public_keys):
  assert isinstance(tensor, (np.ndarray, list, tuple)), type(tensor)
  return convert_to_tensor(tensor, secret_key, public_keys)


def _convert_numpy_tensor(tensor, secret_key, public_keys):
  if len(tensor.shape) > 2:
    raise ValueError("Only matrices are supported for now.")

  # make sure we have a full matrix
  while len(tensor.shape) < 2:
    tensor = np.expand_dims(tensor, 0)

  if np.issubdtype(tensor.dtype, np.float32) \
     or np.issubdtype(tensor.dtype, np.float64):
    # supported as-is
    return Tensor(ops.seal_encrypt(tensor, public_keys), secret_key, public_keys)

  raise ValueError("Don't know how to convert NumPy tensor with dtype '{}'".format(tensor.dtype))


def _convert_tensorflow_tensor(tensor, secret_key, public_keys):
  if len(tensor.shape) > 2:
    raise ValueError("Only matrices are supported for now.")

  # make sure we have a full matrix
  while len(tensor.shape) < 2:
    tensor = tf.expand_dims(tensor, 0)

  if tensor.dtype in (tf.float32, tf.float64):
    # supported as-is
    return Tensor(ops.seal_encrypt(tensor, public_keys), secret_key, public_keys)

  raise ValueError("Don't know how to convert TensorFlow tensor with dtype '{}'".format(tensor.dtype))


def convert_to_tensor(tensor, secret_key, public_keys):
  if isinstance(tensor, Tensor):
    return tensor

  if tensor is None:
    return None

  if isinstance(tensor, (float)):
    return _convert_numpy_tensor(np.array([tensor]), secret_key, public_keys)

  if isinstance(tensor, (list, tuple)):
    return _convert_numpy_tensor(np.array(tensor), secret_key, public_keys)

  if isinstance(tensor, np.ndarray):
    return _convert_numpy_tensor(tensor, secret_key, public_keys)

  if isinstance(tensor, tf.Tensor):
    return _convert_tensorflow_tensor(tensor, secret_key, public_keys)

  raise ValueError("Don't know how to convert value of type {}".format(type(tensor)))


def convert_from_tensor(value, dtype=None):
  assert isinstance(value, Tensor), type(value)

  if dtype is None:
    dtype = tf.float64

  if dtype in [tf.float32, tf.float64]:
    return ops.seal_decrypt(value._raw, value._secret_key, dtype=dtype)

  raise ValueError("Don't know how to evaluate to dtype '{}'".format(dtype))

def add(x, y):
  # TODO(Morten) lifting etc
  return x + y

def sub(x, y):
  # TODO(Morten) lifting etc
  return x - y

def mul(x, y):
  # TODO(Morten) lifting etc
  return x * y

def matmul(x, y):
  # TODO(Morten) lifting etc
  return x.matmul(y)

def poly_eval(x, coeffs):
  res = ops.seal_poly_eval(x._raw, coeffs, x._public_keys)
  return Tensor(res, x._secret_key, x._public_keys)
