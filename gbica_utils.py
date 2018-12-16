import tensorflow as tf
import numpy as np
def estimator_net_split(h_join, h_marg, Wx1,  b1,  Wx2,  b2,  Wx3,  b3,  Wx4,  b4,  Wx5,  b5,  Wx6,  b6):

    dense1_joint = tf.nn.leaky_relu(tf.matmul(h_join, Wx1) + b1)
    dense2_joint = tf.nn.leaky_relu(tf.matmul(dense1_joint, Wx2) + b2)
    dense3_joint = tf.nn.leaky_relu(tf.matmul(dense2_joint, Wx3) + b3)
    dense4_joint = tf.nn.leaky_relu(tf.matmul(dense3_joint, Wx4) + b4)
    dense5_joint = tf.nn.leaky_relu(tf.matmul(dense4_joint, Wx5) + b5)
    dense6_joint = tf.matmul(dense5_joint, Wx6) + b6


    dense1_marg = tf.nn.leaky_relu(tf.matmul(h_marg, Wx1) + b1)
    dense2_marg = tf.nn.leaky_relu(tf.matmul(dense1_marg, Wx2) + b2)
    dense3_marg = tf.nn.leaky_relu(tf.matmul(dense2_marg, Wx3) + b3)
    dense4_marg = tf.nn.leaky_relu(tf.matmul(dense3_marg, Wx4) + b4)
    dense5_marg = tf.nn.leaky_relu(tf.matmul(dense4_marg, Wx5) + b5)
    dense6_marg = tf.matmul(dense5_marg, Wx6) + b6
    return dense6_joint, dense6_marg

def permute_y(y, N):
    range_bs = list(range(N))
    permuted_bs = list(np.random.permutation(range_bs))
    sd_indices =  tf.constant([permuted_bs])
    sd_indices   = tf.transpose(sd_indices)
    shape        = tf.constant([N,1])
    y_    = tf.scatter_nd(sd_indices, y, tf.shape(y))
    return y_