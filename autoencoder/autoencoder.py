import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data
import os

#降噪自编码器
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out) , minval=low,
                              maxval=high, dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale

        self.weights = dict()
        with tf.name_scope("RawInput"):
            self.x = tf.placeholder(tf.float32, [None, self.n_input])
        with tf.name_scope("NoiseAdder"):
            self.scale = tf.placeholder(tf.float32)
            self.noise_x = self.x + self.scale * tf.random_normal((n_input,))
        with tf.name_scope("Encoder"):
            self.weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
            self.weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
            self.hidden = self.transfer(
                    tf.add(tf.matmul(self.noise_x, self.weights['w1']), self.weights['b1']))
        with tf.name_scope("Reconstruction"):
            self.weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
            self.weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
            self.reconstruction = tf.add(tf.matmul(self.hidden ,self.weights['w2']),
                                         self.weights['b2'])
        with tf.name_scope("Cost"):
            #均方误差和
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2))
        with tf.name_scope("Train"):
               self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print("begin to run session……")

    #在一个批次上训练
    def partial_fit(self, X):
        cost,opt = self.sess.run((self.cost, self.optimizer),
            feed_dict={self.x: X, self.scale:self.training_scale})
        return cost

    #在给定的样本集合上计算损失（用于测试阶段）
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale:self.training_scale})

    #返回自编码器隐含层的输出结果，获得抽象后的高阶特征表示
    def transform(self,X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale:self.training_scale})

    #将隐藏层的高阶特征作为输入,将其重建为原始数据
    def generate(self, hidden=None):
        if hidden == None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden:hidden})

    #整体运行一遍复原过程，包括提取高阶特征以及重建原始数据，输入原始数据，输出复原后的数据
    def reconstruction(self, X):
        return self.sess.run(self.reconstruction ,feed_dict={self.x: X, self.scale:self.training_scale} )

    #获取隐含层的权重
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    #获取隐含层的偏置
    def getBiases(self):
        return self.sess.run(self.weights['b1'])
print('产生AdditiveGaussianNoiseAutoencoder')

AGN_AutoEncoder = AdditiveGaussianNoiseAutoencoder(
    n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),scale=0.01)

print('把计算图写入事件文件，在TensorBoard中查看')
writer = tf.summary.FileWriter(logdir='logs', graph=AGN_AutoEncoder.sess.graph)
writer.close()

#读取数据集
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True )
#使用sklearn.preprocess的数据标准化操作（0均值标准差为1）预处理数据
#首先在训练集上估计均值与方差，然后将其作用到训练集和测试集
def standard_scale( X_train, X_test ):
    preprocessor = prep.StandardScaler().fit( X_train )
    X_train = preprocessor.transform( X_train )
    X_test  = preprocessor.transform( X_test )
    return X_train, X_test

#获取随机block数据的函数：取一个从0到len(data) - batch_size的随机整数
#以这个随机整数为起始索引，抽出一个batch_size的批次的样本
def get_random_block_from_data( data, batch_size ):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index+batch_size)]
#使用标准化操作变换数据集，
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
#定义训练参数
n_samples = int(mnist.train.num_examples) #训练样本的总数
training_epochs = 20    #最大训练轮数，每n_samples / batch_size个批次为1轮
batch_size = 128        #每个批次的样本数量
display_step = 1        #输出训练结果的间隔
#下面开始训练过程，在每一轮epoch训练开始时,将平均损失avg_cost设为0
#计算总共需要的batch数量(样本总数/batch size).这里使用的是有放回抽样，所以
#不能保证每个样本被抽到并参与训练。
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    #在每个batch的循环中，随机抽取一个batch的数据，使用成员函数
    #partial_fit训练这个batch的数据，计算cost,累积到当前回合的平均cost中
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = AGN_AutoEncoder.partial_fit(batch_xs)
        avg_cost += cost / batch_size
    avg_cost /= total_batch
    if epoch % display_step == 0:
        print("epoch : %04d, cost = %.9f" % (epoch+1, avg_cost))
#计算测试集上的cost
print("Total cost : ", str(AGN_AutoEncoder.calc_total_cost(X_test)))

