import tensorflow as tf
import numpy as np
import os
import sys
import tarfile
import csv
from six.moves import urllib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#设置算法超参数
learning_rate_init = 0.001
training_epochs = 10
batch_size = 32
display_step = 10
conv1_kernel_num = 64
conv2_kernel_num = 192
conv3_kernel_num = 384
conv4_kernel_num = 256
conv5_kernel_num = 256
fc1_units_num = 4096
fc2_units_num = 4096

#数据集中输入图像的参数
image_size = 224
image_channel = 3
n_classes = 1000

num_examples_per_epoch_for_train = 1000
num_examples_per_epoch_for_eval = 100

#根据指定的维数返回初始化好的指定名称的权重
def WeightsVariable(shape,name_str,stddev=0.1):
    initial = tf.truncated_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

#根据指定的维数返回初始化好的指定名称的偏置
def BiasesVariable(shape,name_str,init_value=0.0):
    initial = tf.constant(init_value,shape=shape)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

#2维卷积层activation（conv2d+bias）的封装
def Conv2d(x,W,b,stride=1,padding='SAME',activation=tf.nn.relu,act_name='relu'):
    with tf.name_scope('conv2d_bias'):
        y = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)
        y = tf.nn.bias_add(y,b)
    with tf.name_scope(act_name):
        y = activation(y)
    return y

#2维池化层pool的封装
def Pool2d(x,pool=tf.nn.max_pool,k=2,stride=2,padding='SAME'):
    return pool(x,ksize=[1,k,k,1],strides=[1,stride,stride,1],padding=padding)

#全连接层activation（wx+b）的封装
def FullyConnected(x,W,b,activate=tf.nn.relu,act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x,W)
        y = tf.add(y,b)
    with tf.name_scope(act_name):
        y = activate(y)
    return y

#为每一层的激活输出添加汇总节点
def AddActivationSummary(x):
    tf.summary.histogram('/activations',x)
    tf.summary.scalar('/sparsity',tf.nn.zero_fraction(x))

#为所有损失节点添加（滑动平均）标量汇总操作
def AddLossesSummary(losses):
    #计算所有(individual losses)和(total loss)的滑动平均
    loss_averages = tf.train.ExponentialMovingAverage(0.9,name='avg')
    loss_averages_op = loss_averages.apply(losses)
    #为所有(individual losses)和(total loss)绑定标量汇总节点
    #为所有平滑处理过的(individual losses)和(total loss)也绑定标量汇总节点
    for loss in losses:
        #没有平滑过的loss名字后面加上（raw），平滑后的loss使用其原来的名称
        tf.summary.scalar(loss.op.name + '(raw)',loss)
        tf.summary.scalar(loss.op.name + '(avg)',loss_averages.average(loss))
    return loss_averages_op

#打印出每一层的输出张量的shape
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


#前向推断过程
def Inference(images_holder):
    #第一个卷积层activate（conv2d+bias）
    with tf.name_scope('Conv2d_1'):
        weights = WeightsVariable(shape=[11, 11, image_channel,conv1_kernel_num],
                                  name_str='weights',stddev=1e-1)
        biases = BiasesVariable(shape=[conv1_kernel_num],name_str='biases',init_value=0.0)
        conv1_out = Conv2d(images_holder,weights,biases,stride=4, padding='SAME')
        AddActivationSummary(conv1_out)
        print_activations(conv1_out)
    #第一个池化层
    with tf.name_scope('Pool2d_1'):
        pool1_out = Pool2d(conv1_out,pool=tf.nn.max_pool,k=3,stride=2,padding='VALID')
        print_activations(pool1_out)
    #第二个卷积层
    with tf.name_scope('Conv2d_2'):
        weights = WeightsVariable(shape=[5, 5, conv1_kernel_num, conv2_kernel_num],
                                  name_str='weights', stddev=1e-1)
        biases = BiasesVariable(shape=[conv2_kernel_num], name_str='biases', init_value=0.0)
        conv2_out = Conv2d(pool1_out, weights, biases, stride=1, padding='SAME')
        AddActivationSummary(conv2_out)
        print_activations(conv2_out)
    #第二个池化层
    with tf.name_scope('Pool2d_2'):
        pool2_out = Pool2d(conv2_out, pool=tf.nn.max_pool, k=3, stride=2, padding='VALID')
        print_activations(pool2_out)
    #第三个卷积层
    with tf.name_scope('Conv2d_3'):
        weights = WeightsVariable(shape=[3, 3, conv2_kernel_num, conv3_kernel_num],
                                  name_str='weights', stddev=1e-1)
        biases = BiasesVariable(shape=[conv3_kernel_num], name_str='biases', init_value=0.0)
        conv3_out = Conv2d(pool2_out, weights, biases, stride=1, padding='SAME')
        AddActivationSummary(conv3_out)
        print_activations(conv3_out)
    # 第四个卷积层
    with tf.name_scope('Conv2d_4'):
        weights = WeightsVariable(shape=[3, 3, conv3_kernel_num, conv4_kernel_num],
                                  name_str='weights', stddev=1e-1)
        biases = BiasesVariable(shape=[conv4_kernel_num], name_str='biases', init_value=0.0)
        conv4_out = Conv2d(conv3_out, weights, biases, stride=1, padding='SAME')
        AddActivationSummary(conv4_out)
        print_activations(conv4_out)
    # 第五个卷积层
    with tf.name_scope('Conv2d_5'):
        weights = WeightsVariable(shape=[3, 3, conv4_kernel_num, conv5_kernel_num],
                                  name_str='weights', stddev=1e-1)
        biases = BiasesVariable(shape=[conv5_kernel_num], name_str='biases', init_value=0.0)
        conv5_out = Conv2d(conv4_out, weights, biases, stride=1, padding='SAME')
        AddActivationSummary(conv5_out)
        print_activations(conv5_out)
    #第三个池化层
    with tf.name_scope('Pool2d_3'):
        pool5_out = Pool2d(conv5_out, pool=tf.nn.max_pool, k=3, stride=2, padding='VALID')
        print_activations(pool5_out)
    #将二维特征图变换为一维特征向量
    with tf.name_scope('FeatsReshape'):
        features = tf.reshape(pool5_out,[batch_size,-1])
        feats_dim = features.get_shape()[1].value
    #第一个全连接层
    with tf.name_scope('FC1_nonlinear'):
        weights = WeightsVariable(shape=[feats_dim,fc1_units_num],
                                  name_str='weights',stddev=4e-2)
        biases = BiasesVariable(shape=[fc1_units_num],name_str='biases',init_value=0.1)
        fc1_out = FullyConnected(features,weights,biases,activate=tf.nn.relu,act_name='relu')
        AddActivationSummary(fc1_out)
        print_activations(fc1_out)
    #第二个全连接层
    with tf.name_scope('FC2_nonlinear'):
        weights = WeightsVariable(shape=[fc1_units_num, fc2_units_num],
                                  name_str='weights',stddev=4e-2)
        biases = BiasesVariable(shape=[fc2_units_num],name_str='biases', init_value=0.1)
        fc2_out = FullyConnected(fc1_out, weights, biases, activate=tf.nn.relu, act_name='relu')
        AddActivationSummary(fc2_out)
        print_activations(fc2_out)
    #第三个全连接层
    with tf.name_scope('FC3_linear'):
        fc3_units_num = n_classes
        weights = WeightsVariable(shape=[fc2_units_num, fc3_units_num],
                                  name_str='weights', stddev=1.0/fc2_units_num)
        biases = BiasesVariable(shape=[fc3_units_num], name_str='biases', init_value=0.0)
        logits = FullyConnected(fc2_out, weights, biases, activate=tf.identity, act_name='linear')
        AddActivationSummary(logits)
        print_activations(logits)
    return logits


#生成假数据用于模型训练过程
def get_faked_train_batch(batch_size):
    images = tf.Variable(tf.random_normal(shape=[batch_size, image_size, image_size, image_channel],
                                          mean=0.0, stddev=1.0, dtype=tf.float32))
    labels = tf.Variable(tf.random_uniform(shape=[batch_size], minval=0, maxval=n_classes,
                                          dtype=tf.int32))
    return images, labels

#生成假数据用于模型测试过程
def get_faked_test_batch(batch_size):
    images = tf.Variable(tf.random_normal(shape=[batch_size, image_size, image_size, image_channel],
                                          mean=0.0, stddev=1.0, dtype=tf.float32))
    labels = tf.Variable(tf.random_uniform(shape=[batch_size], minval=0, maxval=n_classes,
                                          dtype=tf.int32))
    return images, labels

#调用上面写的函数构造计算图
with tf.Graph().as_default():
    #计算图输入
    with tf.name_scope('Inputs'):
        images_holder = tf.placeholder(tf.float32,[batch_size,image_size,image_size,image_channel],name='images')
        labels_holder = tf.placeholder(tf.int32,[batch_size],name='labels')
    #计算图前向推断过程
    with tf.name_scope('Inference'):
        logits = Inference(images_holder)
    #定义损失层（loss layer）
    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_holder,logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy,name='xentropy_loss')
        total_loss_op = cross_entropy_mean
        #average_loss_op = AddLossesSummary(total_loss_op)
    #定义优化训练层(train layer)
    with tf.name_scope('Train'):
        learning_rate = tf.placeholder(tf.float32)
        global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(total_loss_op,global_step=global_step)
    #定义模型评估层（evaluate layer）
    with tf.name_scope('Evaluate'):
        top_K_op = tf.nn.in_top_k(predictions=logits,targets=labels_holder,k=1)

    # 定义获取训练样本批次的计算节点(有数据增强distorted)
    with tf.name_scope('GetTrainBatch'):
        images_train, labels_train = get_faked_train_batch(batch_size=batch_size)
        tf.summary.image('images', images_train, max_outputs=9)
    # 定义获取测试样本批次的节点
    with tf.name_scope('GetTestBatch'):
        images_test, labels_test = get_faked_test_batch(batch_size=batch_size)
        tf.summary.image('images', images_test, max_outputs=9)

    # 收集所有汇总节点
    merged_summaries = tf.summary.merge_all()

    #添加所有变量的初始化节点
    init_op = tf.global_variables_initializer()

    print('在TensorBoard里面查看')
    summary_writer = tf.summary.FileWriter(logdir='logs/alexnet')
    summary_writer.add_graph(graph=tf.get_default_graph())
    summary_writer.flush()
    # graph_writer = tf.summary.FileWriter(logdir='logs/alexnet',graph=tf.get_default_graph())
    # graph_writer.close()
#将评估结果保存到csv文件
    results_list = list()
    #写入参数配置
    results_list.append(['learning_rate',learning_rate_init,
                         'training_epochs',training_epochs,
                         'batch_size',batch_size,
                         'display_step',display_step,
                         'conv1_kernel_num', conv1_kernel_num,
                         'conv2_kernel_num', conv2_kernel_num,
                         'conv3_kernel_num', conv3_kernel_num,
                         'conv4_kernel_num', conv4_kernel_num,
                         'conv5_kernel_num', conv5_kernel_num,
                         'fc1_units_num',fc1_units_num,
                         'fc2_units_num',fc2_units_num])
    #添加表头
    results_list.append(['train_step','train_loss','train_step','train_accuracy'])

    with tf.Session() as sess:
        sess.run(init_op)
        # 启动数据读取队列
        print('======>>>>>>>>>>==开始在训练集上训练模型==<<<<<<<<<<<======')
        num_examples_per_epoch = int(num_examples_per_epoch_for_train / batch_size)
        print("Per batch Size:", batch_size)
        print("Train sample Count Per Epoch:", num_examples_per_epoch_for_train)
        print("Total batch Count Per Epoch:", num_examples_per_epoch)
        # 记录模型被训练的步数
        training_step = 0
        # 训练指定轮数，每一轮的训练样本总数为：num_examples_per_epoch_for_train

        for epoch in range(training_epochs):
            # 每一轮都要把所有的batch跑一边
            for batch_idx in range(num_examples_per_epoch):
                # 运行获取训练数据的计算图，取出一个批次数据
                images_batch, labels_batch = sess.run([images_train, labels_train])
                # 运行优化器训练节点
                _, loss_value = sess.run([train_op, total_loss_op],
                                                     feed_dict={images_holder: images_batch,
                                                                labels_holder: labels_batch,
                                                                learning_rate: learning_rate_init})
                # 每调用一次训练节点，training_step就加1，最终==training_epochs*total_batch
                training_step = sess.run(global_step)
                # 每训练display_step次，计算当前模型的损失和分类准确率
                if training_step % display_step == 0:
                    # 运行accuracy节点，计算当前批次的训练样本的准确率
                    predictions = sess.run([top_K_op],
                                           feed_dict={images_holder: images_batch,
                                                      labels_holder: labels_batch})
                    # 当前每个批次上的预测正确的样本量
                    batch_accuracy = np.sum(predictions) * 1.0 / batch_size
                    results_list.append([training_step, loss_value, training_step, batch_accuracy])
                    print("Training Step: ", str(training_step),
                          ",Training Loss= {:.6f}".format(loss_value),
                          ",Training Accuracy= {:.5f}".format(batch_accuracy))
                    results_list.append([training_step, loss_value, training_step, batch_accuracy])

                    # 运行汇总节点
                    summary_str = sess.run(merged_summaries, feed_dict=
                                          {images_holder: images_batch,
                                           labels_holder: labels_batch})
                    summary_writer.add_summary(summary=summary_str, global_step=training_step)
                    summary_writer.flush()
        print("训练完毕!")

        print('======>>>>>>>>>>==开始在测试集上训练模型==<<<<<<<<<<<======')
        num_batches_per_epoch = int(num_examples_per_epoch_for_eval/batch_size)
        print("Per batch Size:", batch_size)
        print("Test sample Count Per Epoch:", num_examples_per_epoch_for_eval)
        print("Total batch Count Per Epoch:", num_batches_per_epoch)
        correct_predicted = 0
        for batchidx in range(num_batches_per_epoch):
            #运行获取测试数据的计算图，取出一个批次测试数据
            images_batch, labels_batch = sess.run([images_test,labels_test])
            #运行accuracy节点，计算当前批次的测试样本的准确率
            predictions = sess.run([top_K_op],
                                   feed_dict={images_holder:images_batch,
                                              labels_holder:labels_batch})
            #累计每个批次上的预测正确的样本量
            correct_predicted += np.sum(predictions)
        #求在所有测试集上的正确率
        accuracy_score = correct_predicted*1.0/num_batches_per_epoch*batch_size
        print("-------------->Accuracy on Test Examples:",accuracy_score)
        results_list.append(['Accuracy on Test Examples: ',accuracy_score])

    #将评估结果保存到文件
    results_file = open('evaluate_results.csv','w')
    csv_writer = csv.writer(results_file,dialect='excel')
    for row in results_list:
        csv_writer.writerow(row)