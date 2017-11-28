from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
logs_path = '/tmp/tensorflow_logs/example_advanced'

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

x = tf.placeholder("float", [None, n_input], name='InputData')
y = tf.placeholder("float", [None, n_classes], name='LabelData')

with tf.name_scope('ReluLayer1'):
    weights_h1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='weights_h1')
    biases_b1 = tf.Variable(tf.random_normal([n_hidden_1]), name='biases_b1')
    Matmul_1 = tf.matmul(x, weights_h1)
    BiasAdd_1 = tf.add(Matmul_1, biases_b1)
    ReLu_1 = tf.nn.relu(BiasAdd_1)

with tf.name_scope('ReluLayer2'):
    weights_h2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='weights_h2')
    biases_b2 = tf.Variable(tf.random_normal([n_hidden_2]), name='biases_b2')
    Matmul_2 = tf.matmul(ReLu_1, weights_h2)
    BiasAdd_2 = tf.add(Matmul_2, biases_b2)
    ReLu_2 = tf.nn.relu(BiasAdd_2)

with tf.name_scope('LogitLayer'):
    weights_out = tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='weights_out')
    biases_out = tf.Variable(tf.random_normal([n_classes]), name='biases_out')
    Matmul_3 = tf.matmul(ReLu_2, weights_out)
    BiasAdd_3 = tf.add(Matmul_3, biases_out)

with tf.name_scope('softmax'):
    pred = tf.nn.softmax(BiasAdd_3)

with tf.name_scope('CrossEntropy'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=BiasAdd_3, labels=y))

with tf.name_scope('SGD'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads = tf.gradients(cost, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

init = tf.global_variables_initializer()
tf.summary.scalar('Cross Entropy', cost)
tf.summary.scalar('Accuracy', acc)

for grad, var in grads:
    tf.summary.histogram(var.name + '/gradient/', grad)

merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c, summary = sess.run([apply_grads, cost, merged_summary_op], feed_dict={x: batch_x, y: batch_y})
            summary_writer.add_summary(summary, epoch * total_batch +i)
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Run the command line:/n" "--> tensorboard --logdir=/tmp/tensorflow_logs/example_advanced " "/nThen open http://0.0.0.:6006/ into your web browser")