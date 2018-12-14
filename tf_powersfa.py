import tensorflow as tf
import numpy as np

def single_power_step(A, x):
    x = tf.matmul(A, x)
    x = tf.div(x, tf.norm(x))
    return x

def alt_matrix_power(A, x, power):
    with tf.name_scope("matrix_power"):
        iter_count_tf = tf.constant(0)
        condition  = lambda it, A, x: tf.less(it, power)
        body = lambda it, A, x: (it+1, A, single_power_step(A, x))
        loop_vars = [iter_count_tf, A, x]
        output = tf.while_loop(condition, body, loop_vars)[2]
        e = tf.norm(tf.matmul(A, output))
    return output, e

def alt_power_whitening(input_tensor, output_dim, n_iterations=50, **kwargs):
    """
    """
    with tf.name_scope("power_whitening"):
        R = tf.get_variable("randomwhiteningvectors",
                            initializer= np.random.normal(size=(output_dim, output_dim)).astype(np.float32),
                            trainable = False)
        approx_W = tf.get_variable("whiteningmatrix",
                                   initializer=np.zeros(shape=(output_dim, output_dim)).astype(np.float32),
                                   trainable=False)
        input_mean, _ = tf.nn.moments(input_tensor, axes=[0])
        input_tensor = input_tensor - input_mean[None, :]
        C = tf.div(tf.matmul(input_tensor, input_tensor, True, False), tf.cast(tf.shape(input_tensor)[0], tf.float32))
        iter_count_tf = tf.constant(0)
        condition = lambda it, C, W, R: tf.less(it, output_dim)
        def body(it, C, W, R):
            v, l = alt_matrix_power(C, R[:, it, None], n_iterations)
            return (it+1,
                    C - l * tf.matmul(v, v, False, True),
                    W + 1 / tf.sqrt(l) * tf.matmul(v, v, False, True),
                    R)
        approx_W = tf.while_loop(condition,
                                 body,
                                 [iter_count_tf, C, approx_W, R])[2]
        whitened_output = tf.matmul(input_tensor, approx_W, False, True)
    return whitened_output, approx_W, input_mean, C