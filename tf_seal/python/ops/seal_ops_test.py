import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

from tf_seal.python.ops.seal_ops import seal_encrypt
from tf_seal.python.ops.seal_ops import seal_decrypt
from tf_seal.python.ops.seal_ops import seal_encode
from tf_seal.python.ops.seal_ops import seal_decode
from tf_seal.python.ops.seal_ops import seal_add
from tf_seal.python.ops.seal_ops import seal_add_plain

class SealTest(test.TestCase):
  """SealTest test"""

  def test_encrypt_decrypt(self):
    with tf.Session() as sess:
      inp = np.array([[44.44, 55.55], [66.66, 77.77]], np.float32)
      variant = seal_encrypt(inp)
      ans = seal_decrypt(variant, tf.float32)

      np.testing.assert_equal(sess.run(ans), inp)

  def test_encode_decode(self):

  def test_add(self):

  def test_add_plain(self):

if __name__ == '__main__':
  test.main()
