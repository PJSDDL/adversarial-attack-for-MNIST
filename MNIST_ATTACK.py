import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt
import numpy as np

#载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
#设置批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples//batch_size

#定义初始化权值函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

#定义初始化偏置函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#输入层
#定义四个placeholder，前两个输入数据集，后两个输入数字识别网络参数
x = tf.placeholder(tf.float32, [None, 784])  #28*28
y = tf.placeholder(tf.float32, [None, 10])
W_classifi = tf.placeholder(tf.float32, [784, 10])     #28*28*10
b_classifi = tf.placeholder(tf.float32, [10])     

#全连接网络，生成一个28*1的干扰色块
W = weight_variable([28*28, 28*3])
#W = tf.stop_gradient(W)  #加上该语句后，干扰图像不随输入图像变化
b = bias_variable([28*3])
attack = 0.2 * tf.nn.sigmoid(tf.matmul(x, W) + b) 


#利用图像识别网络识别干扰后的图像
attack_img = tf.concat([x[:, 0: 280], attack, x[:, 364: 784]], axis = 1)
prediction = tf.nn.softmax(tf.matmul(attack_img, W_classifi) + b_classifi)


#交叉熵代价函数，目标是使识别准确率尽可能低
cross_entropy = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))

#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#结果存放在一个布尔列表中(argmax函数返回一维张量中最大的值所在的位置)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

#求准确率(tf.cast将布尔值转换为float型)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#创建会话
with tf.Session() as sess:
    start_time = time.clock()
    sess.run(tf.global_variables_initializer()) #初始化变量
    W_TRAINED = np.load("W_TRAINED.npy")
    b_TRAINED = np.load("b_TRAINED.npy")
    for epoch in range(100): 
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #plt.imshow(np.reshape(batch_xs,(28,28)))
            #plt.show()
            sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys, W_classifi:W_TRAINED, b_classifi:b_TRAINED}) #进行迭代训练
        #测试数据计算出准确率
        acc=sess.run(accuracy, feed_dict = {x:mnist.test.images, y:mnist.test.labels, W_classifi:W_TRAINED, b_classifi:b_TRAINED})
        print('Iter' + str(epoch) + ',Testing Accuracy='+str(acc))
    end_time=time.clock()
    print('Running time:%s Second'%(end_time-start_time)) #输出运行时间

    #观察被干扰后神经网络识别情况
    [img, pre] = sess.run([attack_img, prediction], feed_dict = {x:mnist.test.images, y:mnist.test.labels, W_classifi:W_TRAINED, b_classifi:b_TRAINED})

    plt.subplot(251)
    plt.imshow(img[0].reshape(28, 28))
    plt.subplot(256)
    plt.imshow(pre[0].reshape(1, 10))
    plt.subplot(252)
    plt.imshow(img[1].reshape(28, 28))
    plt.subplot(257)
    plt.imshow(pre[1].reshape(1, 10))
    plt.subplot(253)
    plt.imshow(img[2].reshape(28, 28))
    plt.subplot(258)
    plt.imshow(pre[2].reshape(1, 10))
    plt.subplot(254)
    plt.imshow(img[3].reshape(28, 28))
    plt.subplot(259)
    plt.imshow(pre[3].reshape(1, 10))
    plt.subplot(255)
    plt.imshow(img[4].reshape(28, 28))
    plt.subplot(2, 5, 10)
    plt.imshow(pre[4].reshape(1, 10))
    plt.show()
