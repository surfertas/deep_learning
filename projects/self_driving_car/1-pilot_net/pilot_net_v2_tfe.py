# @author Tasuku Miura
# @brief Eager implementation of PilotNet

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

from data_pipeline import *


class PilotNet(tfe.Network):

    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv1 = self.track_layer(
            tf.layers.conv2d(24, [5, 5], (2, 2), "valid", activation=tf.nn.relu))
        self.conv2 = self.track_layer(
            tf.layers.conv2d(36, [5, 5], (2, 2), "valid", activation=tf.nn.relu))
        self.conv3 = self.track_layer(
            tf.layers.conv2d(48, [5, 5], (2, 2), "valid", activation=tf.nn.relu))
        self.conv4 = self.track_layer(
            tf.layers.conv2d(64, [3, 3], (1, 1), "valid", activation=tf.nn.relu))
        self.conv5 = self.track_layer(
            tf.layers.conv2d(64, [3, 3], (1, 1), "valid", activation=tf.nn.relu))

        self.fc1 = self.track_layer(tf.layers.dense(out, 100, tf.nn.relu))
        self.fc2 = self.track_layer(tf.layers.dense(out, 50, tf.nn.relu))
        self.fc3 = self.track_layer(tf.layers.dense(out, 10, tf.nn.relu))
        self.fc4 = self.track_layer(tf.layers.dense(out, 1))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = tf.layers.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


# Helper methods
def mse_loss(predictions, labels):
    return tf.losses.mean_squared_error(
        labels=labels, predictions=predictions)


def train_one_epoch(model, loss, optimizer, dataset, log_interval=None):
    tf.train.get_or_create_global_step()
    for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            with tfe.GradientTape() as tape:
                prediction = model(images, training=True)
                loss_value = loss(prediction, labels)
                tf.contrib.summary.scalar('loss', loss_value)
            grads = tape.gradient(loss_value, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
            if log_interval and batch % log_interval == 0:
                print('Batch #%d\tLoss: %.6f' % (batch, loss_value))


def validate(model, loss, dataset):
    """Perform an evaluation of `model` on the examples from `dataset`."""
    avg_loss = tfe.metrics.Mean('loss')

    for (images, labels) in tfe.Iterator(dataset):
        predictions = model(images, training=False)
        avg_loss(loss(predictions, labels))
        print('Test set: Average loss: %.4f\n' % (avg_loss.result()))
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', avg_loss.result())


def main(_):
    tfe.enable_eager_execution()
    (device, data_format) = ('/gpu:0', 'channels_last')
    print('Using device %s, and data format %s.' % (device, data_format))

    # Set up writers
    if FLAGS.output_dir:
        train_dir = os.path.join(FLAGS.output_dir, 'train')
        valid_dir = os.path.join(FLAGS.output_dir, 'eval')
        tf.gfile.MakeDirs(FLAGS.output_dir)
    else:
        train_dir = None
        valid_dir = None

    summary_writer = tf.contrib.summary.create_file_writer(
        train_dir, flush_millis=10000)
    valid_summary_writer = tf.contrib.summary.create_file_writer(
        valid_dir, flush_millis=10000, name='valid')
    checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, 'ckpt')

    # Set up data
    train_data = DataHandler(
        data_dir=FLAGS.data_dir,
        file_name='train_interpolated.csv',
        bags=config['bags'],
        batch_size=FLAGS.batch_size,
        train=True,
        augment=True
    )
    train_ds = train_data.get_data()

    valid_data = DataHandler(
        data_dir=FLAGS.data_dir,
        file_name='valid_interpolated.csv',
        bags=config['bags'],
        batch_size=1,
        train=False,
        augment=False
    )
    valid_ds = valid_data.get_data()

    # Set up model and optimizer
    model = PilotNet()
    optimizer = tf.train.AdamOptimizer()

    # Train and validate
    with tf.device(device):
        for epoch in range(FLAGS.epochs):
            with tfe.restore_variables_on_create(
                    tf.train.latest_checkpoint(FLAGS.checkpoint_dir)):
                global_step = tf.train.get_or_create_global_step()
                start = time.time()
                with summary_writer.as_default():
                    train_one_epoch(model, mse_loss, optimizer, train_ds, FLAGS.log_interval)
                    end = time.time()
                    print('\nTrain time for epoch #%d (global step %d): %f' % (
                        epoch, global_step.numpy(), end - start))
                with test_summary_writer.as_default():
                    validate(model, test_ds)
                all_variables = (
                    model.variables
                    + optimizer.variables()
                    + [global_step])
                tfe.Saver(all_variables).save(
                    checkpoint_prefix, global_step=global_step)


def __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/input_data',
        help='Directory for storing input data')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        metavar='N',
        help='Directory to write TensorBoard summaries')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints/',
        metavar='N',
        help='Directory to save checkpoints in (once per epoch)')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
