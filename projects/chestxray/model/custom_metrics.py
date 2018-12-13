import tensorflow as tf

def f1_score(predictions=None, labels=None, weights=None):
    # https://stackoverflow.com/questions/44764688/custom-metric-based-on-tensorflows-streaming-metrics-returns-nan/44935895
    P, update_op1 = tf.metrics.precision(predictions, labels)
    R, update_op2 = tf.metrics.recall(predictions, labels)
    eps = 1e-5;
    return (2*(P*R)/(P+R+eps), tf.group(update_op1, update_op2))


def multi_label_accuracy(predictions=None, labels=None):
    # https://stackoverflow.com/questions/37746670/tensorflow-multi-label-accuracy-calculation

    correct_pred = tf.equal(tf.round(labels), predictions)
    # TODO: Check if mean and reduce mean are the same
    accuracy, update = tf.metrics.mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), 1))
    return accuracy, update


