from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

op_lib_file = resource_loader.get_path_to_datafile('_seal_ops.so')
seal_ops = load_library.load_op_library(op_lib_file)

seal_key_gen = seal_ops.seal_key_gen
seal_save_publickey = seal_ops.seal_save_publickey
seal_load_publickey = seal_ops.seal_load_publickey
seal_save_secretkey = seal_ops.seal_save_secretkey
seal_load_secretkey = seal_ops.seal_load_secretkey
seal_save_cipher = seal_ops.seal_save_cipher

seal_encrypt = seal_ops.seal_encrypt
seal_decrypt = seal_ops.seal_decrypt

seal_add = seal_ops.seal_add
seal_add_plain = seal_ops.seal_add_plain
seal_mul = seal_ops.seal_mul
seal_mul_plain = seal_ops.seal_mul_plain
seal_mat_mul = seal_ops.seal_mat_mul
seal_mat_mul_plain = seal_ops.seal_mat_mul_plain
seal_poly_eval = seal_ops.seal_poly_eval
