import tensorflow as tf
import tensorflow_hub as hub

# https://tfhub.dev/s?module-type=image-feature-vector
_RESNET_V2_50 = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1'
_NASNET_A = 'https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/1'
_MOBILENET_V2 = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2'

res_module = hub.Module(_RESNET_V2_50)
nas_module = hub.Module(_NASNET_A)
mob_module = hub.Module(_MOBILENET_V2)


def convnet_ensemble(is_training, x, params):
    # transfer learning, feature extraction
    def _bn_norm(x):
        return tf.layers.batch_normalization(x, momentum=params.bn_momentum, training=is_training)


    print("NAME SCOPE", tf.get_default_graph().get_name_scope())
    with tf.variable_scope('ensemble_res'):
        x_res = res_module(x)
    with tf.variable_scope('ensemble_nas'):
        x_nas = nas_module(x)
    with tf.variable_scope('ensemble_mob'):
        x_mob = mob_module(x)


    # Make dimensions consistent so can add element wise.
    with tf.variable_scope('transition_res'):
        x_res = tf.layers.dense(x_res, 2000)
        if params.use_batch_norm:
            x_res = _bn_norm(x_res)

        x_res = tf.nn.leaky_relu(x_res)
    with tf.variable_scope('transition_nas'):
        x_nas = tf.layers.dense(x_nas, 2000)
        if params.use_batch_norm:
            x_nas = _bn_norm(x_nas)

        x_nas = tf.nn.leaky_relu(x_nas)
    with tf.variable_scope('transition_mob'):
        x_mob = tf.layers.dense(x_mob, 2000)
        if params.use_batch_norm:
            x_mob = _bn_norm(x_mob)

        x_mob = tf.nn.leaky_relu(x_mob)

    x = tf.math.add(tf.math.add(x_res, x_nas), x_mob)

    with tf.variable_scope('fc_1'):
        x = tf.layers.dense(x, 1000)
        if params.use_batch_norm:
            x = _bn_norm(x)

        x = tf.nn.leaky_relu(x)
    with tf.variable_scope('fc_2'):
        logits = tf.layers.dense(x, params.num_labels)

    return logits

