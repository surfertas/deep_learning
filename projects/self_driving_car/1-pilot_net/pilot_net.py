import os
import tensorflow as tf
import numpy as np
from config import config


class PilotNet(object):

    def __init__(self, sess, log_dir, ckpt_dir, n_epochs, batch_size):
        self._sess = sess
        self._log_dir = log_dir
        self._ckpt_dir = ckpt_dir
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._build_graph()

    def _model(self, x):
        assert(x[0].shape == (480, 640, 3))
        out = tf.layers.batch_normalization(x)
        out = tf.layers.conv2d(x, 24, [5, 5], (2, 2), "valid", activation=tf.nn.relu)
        out = tf.layers.conv2d(out, 36, [5, 5], (2, 2), "valid", activation=tf.nn.relu)
        out = tf.layers.conv2d(out, 48, [5, 5], (2, 2), "valid", activation=tf.nn.relu)
        out = tf.layers.conv2d(out, 64, [3, 3], (1, 1), "valid", activation=tf.nn.relu)
        out = tf.layers.conv2d(out, 64, [3, 3], (1, 1), "valid", activation=tf.nn.relu)
        out = tf.reshape(out, [-1, 64 * 53 * 73])
        out = tf.layers.dense(out, 100, tf.nn.relu)
        out = tf.layers.dense(out, 50, tf.nn.relu)
        out = tf.layers.dense(out, 10, tf.nn.relu)
        out = tf.layers.dense(out, 1)
        return out

    def _build_graph(self):
        self._inputs = tf.placeholder("float", [None, 480, 640, 3])
        self._targets = tf.placeholder("float", [None, 1])
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self._predict = self._model(self._inputs)
        self._loss = tf.losses.mean_squared_error(labels=self._targets, predictions=self._predict)
        self._train = tf.train.AdamOptimizer().minimize(self._loss, global_step=self._global_step)

    def _train_admin_setup(self):
        # Writers
        self._train_writer = tf.summary.FileWriter(self._log_dir + '/train', self._sess.graph)

        # Saver
        self._saver = tf.train.Saver()

        # Summaries
        tf.summary.scalar("loss", self._loss)
        self._all_summaries = tf.summary.merge_all()

    def train(self, file_path):
        self._train_admin_setup()
        tf.global_variables_initializer().run()

        # Check if there is a previously saved checkpoint
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self._ckpt_dir))
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from: {}".format(ckpt))
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)

        # Train
        dataset = input_fn(file_path)
        dataset = dataset.batch(self._batch_size)
        print("Data loaded...")
        batch_generator = dataset.make_initializable_iterator()
        next_element = batch_generator.get_next()

        for epoch in range(self._n_epochs):
            self._sess.run(batch_generator.initializer)
            epoch_loss = []
            while True:
                try:
                    img_batch, label_batch = self._sess.run(next_element)
                    loss, _ = self._sess.run([self._loss, self._train],
                                             feed_dict={
                        self._inputs: img_batch['image'],
                        self._targets: label_batch}
                    )
                    epoch_loss.append(loss)
                except tf.errors.OutOfRangeError:
                    break

            # Add summary and save checkpoint after every epoch
            s = self._sess.run(self._all_summaries, feed_dict={
                self._inputs: img_batch['image'],
                self._targets: label_batch}
            )
            self._train_writer.add_summary(s, global_step=epoch)
            print("Epoch: {} Loss: {} Data Count: {}".format(epoch, np.mean(epoch_loss), len(epoch_loss)))
            if epoch % 5 == 0:
                self._saver.save(self._sess, self._ckpt_dir, global_step=self._global_step)

        # Need to closer writers
        self._train_writer.close()

    def evaluate(self):
        pass


def input_fn(file_path):
    def _decode_csv(line):
        # tf.decode_csv needs a record_default as 2nd parameter
        data = tf.decode_csv(line, list(np.array([""] * 12).reshape(12, 1)))[-8:-5]
        img_path = home + data[1]
        img_decoded = tf.to_float(tf.image.decode_image(tf.read_file(img_path)))

        # normalize between (-1,1)
        img_decoded = -1.0 + 2.0 * img_decoded / 255.0
        steer_angle = tf.string_to_number(data[2], tf.float32)

        # return camera as well to confirm location of camera
        # e.g left_camera, center_camera, right_camera...we want center_camera
        return {'image': img_decoded, 'camera': data[0]}, [steer_angle]

    # skip() header row,
    # filter() for center camera,
    # map() transform each line by applying decode_csv()
    dataset = (tf.contrib.data.TextLineDataset(file_path)
               .skip(1)
               .filter(lambda x: tf.equal(
                       tf.string_split(tf.reshape(x, (1,)), ',').values[4],
                       'center_camera'))
               .map(_decode_csv))
    return dataset


if __name__ == "__main__":
    home = config['bag4']
    file_path = home + "interpolated.csv"
    tf.reset_default_graph()

    # Want to see what devices are being used.
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        model = PilotNet(sess, 'pilot_net', 'checkpoints/pilot_net', 10, 32)
        model.train(file_path)
