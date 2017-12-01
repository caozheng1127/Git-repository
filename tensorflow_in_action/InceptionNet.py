import tensorflow as tf
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def inception_v3_arg_scopoe(weight_decay=0.00004, stddev=0.1, batch_norm_var_collection='moving_vars'):
    batch_norm_params = {
        'decay': 0.9997, 'epsilon': 0.001, 'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {'beta': None, 'gamma': None, 'moving_mean': [batch_norm_var_collection], 'moving_variance': [batch_norm_var_collection],}
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=stddev),activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params) as sc:
            return sc

def inception_v3_base(inputs, scope=None):
    end_points = {}
    with tf.vatiable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
            net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
