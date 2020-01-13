import numpy as np
import tensorflow as tf
import tf_seal as tfs

public_keys, secret_key = tfs.seal_key_gen(gen_relin=True, gen_galois=True)

# sample inputs in the form of tf.Tensors
a = tf.random.normal(shape=(2, 3), dtype=tf.float32)
b = tf.random.normal(shape=(2, 3), dtype=tf.float32)

# the plaintext equivalent of our computation
c = tf.matmul(a, tf.transpose(b))

# encrypt inputs, yielding tf_seal.Tensors
a_encrypted = tfs.convert_to_tensor(a, secret_key, public_keys)
b_encrypted = tfs.convert_to_tensor(b, secret_key, public_keys)

# perform computation on encrypted data
# - note that because of how the data is laid out in memory,
#   tf_seal.matmul expects the right-hand matrix to be ordered
#   column-major wise, i.e. already transposed
c_encrypted = tfs.matmul(a_encrypted, b_encrypted)

with tf.Session() as sess:
    expected, actual = sess.run([c, c_encrypted])
    np.testing.assert_almost_equal(actual, expected, decimal=3)
