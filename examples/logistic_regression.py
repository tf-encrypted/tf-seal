import numpy as np
import tensorflow as tf

import tf_seal as tfs

def tf_log_reg(x, y):
    z = tf.matmul(x, y)
    return tf.sigmoid(z)

def sigmoid(x):
    # the following polynomial evaluation approximates by computing
    # 0.5 + 0.197x + -0.004x^3, the 0.0 term is ignored. This
    # approximation was borrowed from https://eprint.iacr.org/2018/462 Section 3.2 Fig 1
    coeffs = np.array([0.5, 0.197, 0.0, -0.004])
    return tfs.poly_eval(x, coeffs)

public_keys, secret_key = tfs.seal_key_gen(gen_relin=True, gen_galois=True)

# encrypted input -> tf_seal.Tensor
a_plain = np.random.normal(size=(20, 20)).astype(np.float32)
a = tfs.constant(a_plain, secret_key, public_keys)
# public weights
b = np.random.normal(size=(20, 20)).astype(np.float32)

c = tfs.matmul(a, b.transpose())
#print("Here")
d = sigmoid(c)
print(d)
# get answer from tensorflow to compare
tf_d = tf_log_reg(a_plain, b)

with tf.Session() as sess:
    print("TF SEAL Logistic Regression")
    print(sess.run(d))
    print("\nApproximately Equals\n")
    print("Tensorflow Logistic Regression")
    print(sess.run(tf_d))
