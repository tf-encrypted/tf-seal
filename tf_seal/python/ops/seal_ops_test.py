import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

from tf_seal.python.ops.seal_ops import seal_encrypt
from tf_seal.python.ops.seal_ops import seal_decrypt

class SealTest(test.TestCase):
  """SealTest test"""

  def test_import_export(self):
    with tf.Session() as sess:
      inp = np.array([[44.44, 55.55], [66.66, 77.77]], np.float32)
      variant = seal_encrypt(inp)
      ans = seal_decrypt(variant, tf.float32)

      np.testing.assert_equal(sess.run(ans), inp)

if __name__ == '__main__':
  test.main()
