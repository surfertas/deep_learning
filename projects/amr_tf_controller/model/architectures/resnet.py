import tensorflow as tf
import tensorflow_hub as hub

# https://tfhub.dev/s?module-type=image-feature-vector
_RESNET_V2_50 = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1'

module = hub.Module(_RESNET_V2_50)

def resnet_v2_50_tl_fe(is_training, x, params):
    # transfer learning, feature extraction

    print("NAME SCOPE", tf.get_default_graph().get_name_scope())
    with tf.variable_scope('resnet_1'):
        x = module(x)
    with tf.variable_scope('fc_1'):
        x = tf.layers.dense(x, 1000)
        x = tf.nn.leaky_relu(x)
    with tf.variable_scope('fc_2'):
        logits = tf.layers.dense(x, params.num_labels)

    return logits
