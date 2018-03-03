import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#设置算法超参数
learning_rate_init = 0.001
training_epochs = 5
batch_size = 32
display_step = 2
keep_prob = 0.8

#数据集中输入图像的参数
image_size = 224
image_channel = 3
n_classes = 1000
num_examples_per_epoch_for_train = 2000
num_examples_per_epoch_for_eval = 1000

#根据指定的维数返回初始化好的指定名称的权重
def WeightsVariable(shape,name_str,stddev=0.1):
    initial = tf.truncated_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

#根据指定的维数返回初始化好的指定名称的偏置
def BiasesVariable(shape,name_str,init_value=0.0):
    initial = tf.constant(init_value,shape=shape)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

#2维卷积层activation（conv2d+bias）的封装
def Conv2d_op(input_op, name, kh, kw, n_out, dh, dw,
              activation_func=tf.nn.relu, activation_name='relu'):
    with tf.name_scope(name) as scope:
        n_in = input_op.get_shape()[-1].value
        kernals = tf.get_variable(scope+'weight', shape=[kh,kw,n_in,n_out],dtype =tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernals, strides=(1, dh ,dw, 1), padding = 'SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='bias')
        z = tf.nn.bias_add(conv,biases)
        activation = activation_func(z, name=activation_name)
        return activation

#2维池化层pool的封装
def Pool2d_op(input_op, name, kh=2, kw=2,dh=2,dw=2,padding='SAME',pool_func=tf.nn.max_pool):
    with tf.name_scope(name) as scope:
        return pool_func(input_op,
                         ksize=[1,kh,kw,1],strides=[1,dh,dw,1],
                         padding=padding, name=name)

#全连接层activation（wx+b）的封装
def FullyConnected(input_op, name, n_out, activation_func=tf.nn.relu,activation_name='relu'):
    with tf.name_scope(name) as scope:
        n_in = input_op.get_shape()[-1].value
        kernals = tf.get_variable(scope + 'weight', shape=[ n_in, n_out], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        bias_init_val = tf.constant(0.1, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='bias')
        z = tf.nn.bias_add(tf.matmul(input_op, kernals), biases)
        activation = activation_func(z, name=activation_name)
        return activation

#为每一层的激活输出添加汇总节点
def AddActivationSummary(x):
    tf.summary.histogram('/activations', x)
    tf.summary.scalar('/sparsity', tf.nn.zero_fraction(x))

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
def Inference(images_holder,keep_prob):
    #第一段卷积网络
    conv1_1 = Conv2d_op(images_holder, name='conv1_1',kh=3,kw=3,n_out=64,dh=1,dw=1)
    pool1 = Pool2d_op(conv1_1,name='pool1',kh=2,kw=2,dh=2,dw=2,padding='SAME')
    print_activations(pool1)
    #第二段卷积网络
    conv2_1 = Conv2d_op(pool1, name='conv2_1',kh=3,kw=3,n_out=128,dh=1,dw=1)
    pool2 = Pool2d_op(conv2_1,name='pool2',kh=2,kw=2,dh=2,dw=2,padding='SAME')
    print_activations(pool2)
    #第三段卷积网络
    conv3_1 = Conv2d_op(pool2, name='conv3_1',kh=3,kw=3,n_out=256,dh=1,dw=1)
    conv3_2 = Conv2d_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1)
    pool3 = Pool2d_op(conv3_2,name='pool3',kh=2,kw=2,dh=2,dw=2,padding='SAME')
    print_activations(pool3)
    #第四段卷积网络
    conv4_1 = Conv2d_op(pool3, name='conv4_1',kh=3,kw=3,n_out=512,dh=1,dw=1)
    conv4_2 = Conv2d_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool4 = Pool2d_op(conv4_2,name='pool4',kh=2,kw=2,dh=2,dw=2,padding='SAME')
    print_activations(pool4)
    #第五段卷积网络
    conv5_1 = Conv2d_op(pool4, name='conv5_1',kh=3,kw=3,n_out=512,dh=1,dw=1)
    conv5_2 = Conv2d_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool5 = Pool2d_op(conv5_2,name='pool5',kh=2,kw=2,dh=2,dw=2,padding='SAME')
    print_activations(pool5)

    #将二维特征图变换为一维特征向量
    with tf.name_scope('FeatReshape'):
        features = tf.reshape(pool5, [batch_size, -1])
        feats_dim = features.get_shape()[1].value

    #第一个全连接层
    fc1_out = FullyConnected(features, name='fc1', n_out=4096,
                             activation_func=tf.nn.relu, activation_name='relu')
    print_activations(fc1_out)
    #dropout
    with tf.name_scope('dropout_1'):
        fc1_dropout = tf.nn.dropout(fc1_out, keep_prob=keep_prob)
    # 第二个全连接层
    fc2_out = FullyConnected(fc1_dropout, name='fc2', n_out=4096,
                             activation_func=tf.nn.relu, activation_name='relu')
    print_activations(fc2_out)
    # dropout
    with tf.name_scope('dropout_2'):
        fc2_dropout = tf.nn.dropout(fc2_out, keep_prob=keep_prob)
    # 第三个全连接层
    logits = FullyConnected(fc2_dropout, name='fc3', n_out=1000,
                             activation_func=tf.identity, activation_name='identity')
    print_activations(logits)
    return logits

def get_distorted_train_batch(data_dir,batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    images,labels = cifar_input.distorted_inputs(data_dir=data_dir,
                                                 batch_size=batch_size)
    return images,labels

def get_undistorted_eval_batch(data_dir,eval_data,batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    images,labels = cifar_input.inputs( eval_data=True,
                                         data_dir=data_dir,
                                         batch_size=batch_size)
    return images,labels

def TrainAndTestModel():
    with tf.Graph().as_default():
        with tf.name_scope('Inputs'):
            images_holder = tf.placeholder(tf.float32,
                                           [batch_size, image_size, image_size, image_channel],
                                           name='images')
            labels_holder = tf.placeholder(tf.int32, [batch_size], name='labels')
        # 计算图前向推断过程
        with tf.name_scope('Inference'):
            keep_prob_holder = tf.placeholder(tf.float32, name='KeepProb')
            logits = Inference(images_holder, keep_prob_holder)
        # 定义损失层（loss layer）
        with tf.name_scope('Loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_holder, logits=logits)
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            total_loss_op = cross_entropy_mean
        # 定义优化训练层(train layer)
        with tf.name_scope('Train'):
            learning_rate = tf.placeholder(tf.float32)
            global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(total_loss_op, global_step=global_step)
        # 定义模型评估层（evaluate layer）
        with tf.name_scope('Evaluate'):
            top_K_op = tf.nn.in_top_k(predictions=logits, targets=labels_holder, k=1)
        # 定义获取训练样本批次的计算节点(有数据增强distorted)
        with tf.name_scope('GetTrainBatch'):
            images_train, labels_train = get_faked_train_batch(batch_size=batch_size)
            tf.summary.image('images', images_train, max_outputs=9)
        # 定义获取测试样本批次的节点
        with tf.name_scope('GetTestBatch'):
            images_test, labels_test = get_faked_ceshi_batch(batch_size=batch_size)
            tf.summary.image('images', images_test, max_outputs=9)
        #添加所有变量的初始化节点
        init_op = tf.global_variables_initializer()
        print('在TensorBoard里面查看')

        graph_writer = tf.summary.FileWriter(logdir='logs/vggnet',graph=tf.get_default_graph())
        graph_writer.close()

def main(argv=None):
    #创建日志目录
    train_dir = 'logs/'
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)
    #训练并且测试模型
    TrainAndTestModel()

if __name__ == '__main__':
    tf.app.run()
