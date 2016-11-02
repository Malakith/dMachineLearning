import tensorflow as tf
import convNetwork as n
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Import mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist_reshape_first = np.asarray(mnist.train.images)[0, :].reshape(28, 28)


# function to transform data to one_hot format.
def one_hot(vector, max_int=10):
    return np.eye(max_int)[vector]


# First we load the AU digits.
train_file = np.load('auTrain.npz')
images_train = train_file['digits']
labels_train = one_hot(train_file['labels'])
test_file = np.load('auTest.npz')
images_test = test_file['digits']
labels_test = test_file['labels']

# Then we add the mnist data:

images_train = np.concatenate((mnist.train.images, images_train), axis=0)
labels_train = np.concatenate((mnist.train.labels, labels_train), axis=0)

print(images_train.shape)
print(labels_train.shape)

print(images_train.dtype)
print(labels_train.dtype)

# Here we can set our training variables:

filename = n.getFilename()

epochs = n.getEpochs()
learning_rate = n.getLearningRate()
batch_size = n.getBatchSize()
keep_prop = n.getKeepProp()

results = list()

# Now we begin training

def train(images, labels, learning_rate, keep_prop, batch_size, epochs, cont = 0):
    with tf.Graph().as_default():
        n, d = images.shape
        max_step = (n / batch_size) * epochs
        images_ph, labels_ph, learning_rate_ph, keep_prop_ph, init_op, train_op, pred_op, eval_op, saver = \
            n.buildModel(images, labels, keep_prop, batch_size, epochs, learning_rate)

        feed_dict = {images_ph: images, labels_ph: labels, learning_rate_ph: learning_rate, keep_prop_ph: keep_prop}
        with tf.Session() as session:
            session.run(init_op, feed_dict=feed_dict)
            step = 0
            if cont > 0:
                saver.restore(sess=session, save_path=filename + "_step_" + str(cont))
                step = cont

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            try:
                while not coord.should_stop():
                    _, loss_value = session.run(train_op)
                    print("\rCompleted step " + str(step), end="")
                    if step % 100 == 0:
                        saver.save(session, filename + "_step_" + str(step))
                        eval_value = session.run(eval_op, feed_dict={keep_prop_ph: 1.0})
                        in_sample_acc = (eval_value/batch_size)*100
                        percent_done = (step / max_step) * 100.0
                        pred = predict(images_test, step)
                        out_of_sample_acc = (np.sum(pred == labels_test)/images_test.shape[0])*100
                        print("\n\n\nCompleted %.2f percent / %d steps. \n"
                              "In sample accuracy is %.2f%% and loss is %.4f.\n"
                              "Out of sample accuracy is %.2f%%\n\n\n" % (
                              percent_done, step, in_sample_acc, loss_value, out_of_sample_acc))


                        results.append([step, in_sample_acc, out_of_sample_acc])
                        np.save(filename, np.asarray(results))

                    step += 1
            except tf.errors.OutOfRangeError:
                saver.save(session, filename + "_final")
                print('\rDone training for %d epochs, %d steps.' % (epochs, step))
            finally:
                coord.request_stop()

            coord.join()
            saver.save(session, filename)
            session.close()


def predict(images, step=0):
    n, d = images.shape
    labels = np.zeros(shape=(n, 10))
    learning_rate = 0.0
    keep_prop = 1.0
    batch_size = n
    epochs = 1
    prediction = list()
    with tf.Graph().as_default():

        max_step = (n / batch_size) * epochs
        images_ph, labels_ph, learning_rate_ph, keep_prop_ph, init_op, train_op, pred_op, eval_op, saver = \
            n.buildModel(images, labels, keep_prop, batch_size, epochs, learning_rate, shuffle=False)



        feed_dict = {images_ph: images, labels_ph: labels, learning_rate_ph: learning_rate, keep_prop_ph: keep_prop}
        with tf.Session() as session:
            session.run(init_op, feed_dict=feed_dict)
            saver.restore(sess=session, save_path=filename + "_step_" + str(step))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            step = 0
            try:
                while not coord.should_stop():
                    prediction.append(session.run(pred_op, feed_dict={images_ph: images}))
                    step += 1
            except tf.errors.OutOfRangeError:
                pass
            finally:
                coord.request_stop()

            coord.join()
            session.close()
            return prediction

train(images_train, labels_train, learning_rate, keep_prop, batch_size, epochs)
pred = predict(images_test)
print(np.sum(pred == labels_test)/images_test.shape[0])

