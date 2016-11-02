import tensorflow as tf

filename = "ConvoNetwork"

epochs = 100
learning_rate = 0.01
batch_size = 512
keep_prop = 0.5


def getEpochs(): return epochs
def getLearningRate(): return learning_rate
def getBatchSize(): return batch_size
def getKeepProp(): return keep_prop

def getFilename(): return filename

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def build_placeholders(n, batch_size, epochs, shuffle=True, threads=1):
    images_ph = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='imagesPlaceholder')
    labels_ph = tf.placeholder(tf.float32, shape=[None, 10], name='labelsPlaceholder')
    images = tf.reshape(tf.Variable(images_ph, trainable=False, validate_shape=False, name='images_var'),
                        shape=[-1, 28 * 28], name='images_reshape')
    labels = tf.reshape(tf.Variable(labels_ph, trainable=False, validate_shape=False, name='labels_var'),
                        shape=[-1, 10], name="labels_reshape")
    images, labels = tf.train.slice_input_producer([images, labels], num_epochs=epochs, shuffle=shuffle,
                                                   name='inputslicer')
    images, labels = tf.train.batch([images, labels], batch_size=batch_size,
                                    name='batches', enqueue_many=False, num_threads=threads)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='learning_ratePlaceholder')
    learning_rate = tf.Variable(learning_rate_ph, trainable=False, name='learning_rate_var')
    keep_prop_ph = tf.placeholder(tf.float32, shape=[], name="keep_prop_placeholder")
    keep_prop_ = tf.Variable(keep_prop_ph, trainable=False, name='keep_prop_var')
    return images_ph, images, labels_ph, labels, learning_rate_ph, learning_rate, keep_prop_ph, keep_prop_


def construct_graph(images, keep_prop):
    # Input layer
    image = tf.reshape(images, [-1, 28, 28, 1])
    # First convoluted layer
    w1 = weight_variable([5, 5, 1, 32], name="w1")
    b1 = bias_variable([32], name='b1')
    h_c1 = tf.nn.relu(conv2d(image, w1) + b1)
    h_p1 = max_pool_2x2(h_c1)
    # Second convoluted layer
    w2 = weight_variable([5, 5, 32, 64], name="w2")
    b2 = bias_variable([64], name='b2')
    h_c2 = tf.nn.relu(conv2d(h_p1, w2) + b2)
    h_p2 = max_pool_2x2(h_c2)
    # Densely connected layer
    w3 = weight_variable([7 * 7 * 64, 1024], name="w3")
    b3 = bias_variable([1024], name='b3')
    h_p2_flat = tf.reshape(h_p2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_p2_flat, w3) + b3)
    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prop)
    # Readout layer
    w4 = weight_variable([1024, 10], name="w4")
    b4 = bias_variable([10], name='b4')
    logits = tf.matmul(h_fc1_drop, w4) + b4
    saver = tf.train.Saver({'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3, 'w4': w4, 'b4': b4}
                           , max_to_keep=0)
    return logits, saver


def loss(logits, labels):
    # logits_softmax = tf.clip_by_value(logits, np.nextafter(0,1) ,np.nextafter(1,0))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name="xEntropy")
    loss = tf.reduce_mean(cross_entropy, name="lossValue")
    return loss


def train(loss, learning_rate_start):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate_start, global_step, 100, 0.90, staircase=True,
                                               name="exp_learning_rate")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='AdamOptimizer')
    train_step = optimizer.minimize(loss, global_step=global_step, name="train_step")
    return train_step


def evaluate(logits, labels):
    # Returns the number of correct predictions.
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1), name="correct_predicitons")
    return tf.reduce_sum(tf.cast(correct_prediction, tf.float32), name='evaluation')


def buildModel(images, labels, keep_prop, batch_size, epochs, learning_rate, shuffle=True, threads=1):
    n, d = images.shape
    # We build placeholders for our data and variables.
    images_ph, images_, labels_ph, labels_, learning_rate_ph, learning_rate_, keep_prop_ph, keep_prop_ = \
        build_placeholders(n, batch_size, epochs, shuffle, threads=threads)
    # Now we need to build our graph
    logits, saver = construct_graph(images_, keep_prop_)
    # We need a loss function
    loss_ = loss(logits, labels_)
    # And lastly we need a train op

    train_op = [train(loss_, learning_rate_), loss_]

    pred_op = tf.argmax(logits, 1)

    # And we also want to be able to see how well we classify
    eval_op = evaluate(logits, labels_)

    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    return images_ph, labels_ph, learning_rate_ph, keep_prop_ph, init_op, train_op, pred_op, eval_op, saver
