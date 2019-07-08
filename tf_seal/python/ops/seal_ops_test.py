import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

from tf_seal.python.ops.seal_ops import seal_import

class SealTest(test.TestCase):
  """SealTest test"""

  def test_import_export(self):
    with tf.Session() as sess:
      inp = [[44.44]]
      variant = seal_import(inp)

      variant.eval()