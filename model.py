import tensorflow as tf
import abc
import six
import numpy as np
import tflearn


@six.add_metaclass(abc.ABCMeta)
class ConvModel():
    def __init__(self, inp, out_classes):
        self.input = inp
        self.training = tf.Variable(True, trainable=False)
        self.output = self._build_net(inp, self.training)


    @abc.abstractmethod
    def _build_net(self, inp, training):
        pass

    @abc.abstractproperty
    def name(self):
        raise NotImplementedError("Did not implement name attribute of model")


class SmallConvNet(ConvModel):
    @property
    def name(self):
        return "Small convnet"

    def _build_net(self, inp, training, out_classes=3):
        net = tf.image.resize_images(inp, [226, 226])

        for i in range(5):
            for j in range(3):
                pshape = net.get_shape().as_list()
                net = tf.layers.conv2d(net, 4 * (1<<i), 3, strides=(1, 1),
                        padding="same", name="conv{}_{}".format(i + 1, j + 1),
                        kernel_initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(1.0 / (3 * 3 * pshape[3]))))
                net = selu(net)
                net = dropout_selu(net, 0.5, training=training)
            net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, out_classes, name="fc")

        return net


class DenseNet(ConvModel):
    @property
    def name(self):
        return "densenet"


    def _build_net(self, inp, training, out_classes=3):
        K = 12
        theta = 0.5
        init = tf.random_normal_initializer
        net = tf.image.resize_images(inp, [226, 226])
        net = tf.layers.conv2d(net, 2 * K, 7, 2, "same",
                kernel_initializer=init(stddev=np.sqrt(1.0/(7 * 7))))
        net = selu(net)
        net = tf.layers.max_pooling2d(net, 3, 2)

        l = []
        for i in range(3):
            l = [net]
            for j in range(6):
                with tf.variable_scope("dense{}_{}".format(i+1, j+1)):
                    # Note: players = previous layers
                    players = tf.concat(l, 3)
                    net = tf.layers.conv2d(players, 4 * K, 1, padding="same",
                            kernel_initializer=init(stddev=np.sqrt(1.0 /
                                (players.get_shape().as_list()[3]))),
                            name="conv1x1")
                    net = selu(net)

                    net = tf.layers.conv2d(players, K, 3, padding="same",
                            kernel_initializer=init(
                                stddev=np.sqrt(1.0 / (4 * K * 9))),
                            name="conv3x3")
                    net = selu(net)
                    net = dropout_selu(net, 0.5, training=training)
                    l.append(net)
            if i == 2:
                break
            players = tf.concat(l, 3)
            fan_in = players.get_shape().as_list()[3]
            net = tf.layers.conv2d(players, fan_in // 2, 1, padding="same",
                    kernel_initializer=init(stddev=np.sqrt(1.0/(fan_in))),
                    name="transition{}".format(i+1))
            net = selu(net)
            net = tf.layers.average_pooling2d(net, 2, 2)

        players = tf.concat(l, 3)
        ps = net.get_shape().as_list()[1]
        net = tf.layers.average_pooling2d(players, ps, ps)
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, 3, name="fc")
        return net


class ImageClassifier():
    def __init__(self, M, out_classes):
        self.X = tf.placeholder(tf.float32, [None, 226, 226, 1], name="images")
        tf.summary.image("input_image", self.X)
        self.Y = tf.placeholder(tf.int32, [None], name="labels")
        Y_exp = tf.one_hot(self.Y, out_classes)
        m = M(self.X, out_classes)
        self.m = m

        self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=m.output, labels=Y_exp))
        accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(m.output, axis=1), tf.argmax(Y_exp, axis=1)),
                    tf.float32))
        self.accuracy = accuracy
        tf.summary.scalar("error", self.error)
        tf.summary.scalar("accuracy", accuracy)

        self.global_step = tf.Variable(0, trainable=False, name="global_step")

    @property
    def name(self):
        return self.m.name


def tflearn_conv_model():
    net = tflearn.input_data(shape=[None, 256, 256, 1])
    net = tflearn.conv_2d(net, 32, 7, strides=2)
    net = tflearn.selu(net)
    net = tflearn.max_pool_2d(net, 2)

if __name__ == '__main__':
    X = tf.placeholder(tf.float32, [None, 226, 226, 1])
    m = DenseNet(X, 3)
