"""Define the model."""

import tensorflow as tf

from model.architectures.nasnet import nasnet_tl_fe
from model.architectures.resnet import resnet_v2_50_tl_fe
from model.architectures.ensemble import convnet_ensemble




from model.custom_metrics import f1_score, multi_label_accuracy

def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images']
    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]


    out = images
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    print(out.get_shape().as_list())

    assert out.get_shape().as_list() == [None, 14, 14, num_channels * 8]

    out = tf.reshape(out, [-1, 14 * 14 * num_channels * 8])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, num_channels * 8)
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
    with tf.variable_scope('fc_2'):
        logits = tf.layers.dense(out, params.num_labels)

    return logits

def build_model_0000(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images']
    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]
    out = images
    
    logits = nasnet_tl_fe(is_training, out, params)

    return logits


def build_model_0001(is_training, inputs, params):
    images = inputs['images']
    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]
    out = images

    logits = resnet_v2_50_tl_fe(is_training, out, params)
    return logits

def build_model_0002(is_training, inputs, params):
    images = inputs['images']
    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]
    out = images

    logits = convnet_ensemble(is_training, out, params)
    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    #print("LABELS", labels.get_shape())

    build_model_dict = {
        "base_model": build_model,
        "build_model_0000": build_model_0000,
        "build_model_0001": build_model_0001,
        "build_model_0002": build_model_0002,
    }


    ##http://vict0rsch.github.io/2018/06/17/multilabel-text-classification-tensorflow/
    def _multi_label_hot(prediction, threshold=0.5):
        prediction = tf.cast(prediction, tf.float32)
        threshold = float(threshold)
        return tf.cast(tf.greater(prediction, threshold), tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        #logits = build_model_0001(is_training, inputs, params)
        logits = build_model_dict[params.build_model_version](is_training, inputs, params)

        unrounded_predictions = tf.sigmoid(logits)
        predictions = _multi_label_hot(tf.sigmoid(logits)) # basically rounding to 0 or 1

    # Define loss and accuracy
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

    # https://stackoverflow.com/questions/37746670/tensorflow-multi-label-accuracy-calculation
    correct_pred = tf.equal(tf.round(labels), predictions)

    # true positive is defined as ALL classes labelled correctly
    accuracy = tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), 1))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):

        metrics = {
            "auc roc Atelectasis" : tf.metrics.auc(labels=labels[:, 0], predictions=unrounded_predictions[:, 0]),
            "auc roc Cardiomegaly" : tf.metrics.auc(labels=labels[:, 1], predictions=unrounded_predictions[:, 1]),
            "auc roc Consolidation" : tf.metrics.auc(labels=labels[:, 2], predictions=unrounded_predictions[:, 2]),
            "auc roc Edema" : tf.metrics.auc(labels=labels[:, 3], predictions=unrounded_predictions[:, 3]),
            "auc roc Effusion" : tf.metrics.auc(labels=labels[:, 4], predictions=unrounded_predictions[:, 4]),
            "auc roc Emphysema" : tf.metrics.auc(labels=labels[:, 5], predictions=unrounded_predictions[:, 5]),
            "auc roc Fibrosis" : tf.metrics.auc(labels=labels[:, 6], predictions=unrounded_predictions[:, 6]),
            "auc roc Hernia" : tf.metrics.auc(labels=labels[:, 7], predictions=unrounded_predictions[:, 7]),
            "auc roc Infiltration" : tf.metrics.auc(labels=labels[:, 8], predictions=unrounded_predictions[:, 8]),
            "auc roc Mass": tf.metrics.auc(labels=labels[:, 9], predictions=unrounded_predictions[:, 9]),
            "auc roc Nodule": tf.metrics.auc(labels=labels[:, 10], predictions=unrounded_predictions[:, 10]),
            "auc roc Pleural_Thickening": tf.metrics.auc(labels=labels[:, 11], predictions=unrounded_predictions[:, 11]),
            "auc roc Pneumonia": tf.metrics.auc(labels=labels[:, 12], predictions=unrounded_predictions[:, 12]),
            "auc roc Pneumothorax": tf.metrics.auc(labels=labels[:, 13], predictions=unrounded_predictions[:, 13]),

            'accuracy': multi_label_accuracy(labels=labels, predictions=predictions),
            'f1_score': f1_score(labels=labels, predictions=predictions),
            'recall': tf.metrics.recall(labels=labels, predictions=predictions),
            'precision': tf.metrics.precision(labels=labels, predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }


    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    # mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    # for label in range(0, params.num_labels):
    #     mask_label = tf.logical_and(mask, tf.equal(predictions, label))
    #     incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
    #     tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy   
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
