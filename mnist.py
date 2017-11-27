from __future__ import print_function
# 导入MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("", one_hot=True)
import tensorflow as tf
# 参数的设置
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
# 网络参数
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
# tf Graph的输入
x = tf.placeholder("float", [None, n_input]) # 用placeholder先占地方，样本个数不确定为None
y = tf.placeholder("float", [None, n_classes]) # 用placeholder先占地方，样本个数不确定为None

# 初始化weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# 创建模型
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #输出层使用线性激活函数
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# 构建模型
pred = multilayer_perceptron(x, weights, biases)
# 定义损失函数和优化
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# 初始化变量
init = tf.global_variables_initializer()
# 定义一个Session
with tf.Session() as sess:
    sess.run(init)
# 循环训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)# 分批训练
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
# 数据的喂给，并运行optimizer和cost
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
#计算平均损失函数
            avg_cost += c / total_batch
# 每epoch step 显示log日志
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
    print("Optimization Finished!")
# 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# 计算精确度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))