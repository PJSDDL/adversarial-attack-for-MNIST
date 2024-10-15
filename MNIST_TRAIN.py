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
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#定义初始化偏置函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#输入层
#定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784]) #28*28
y = tf.placeholder(tf.float32, [None, 10])

#全连接
W = weight_variable([28*28, 10])
b = bias_variable([10])

#输出层
#计算输出
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

#交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

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
    for epoch in range(40):     
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            #plt.imshow(np.reshape(batch_xs,(28,28)))
            #plt.show()
            sess.run(train_step, feed_dict={x:batch_xs,y:batch_ys}) #进行迭代训练
        #测试数据计算出准确率
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('Iter' + str(epoch) + ',Testing Accuracy='+str(acc))
    end_time = time.clock()
    print('Running time:%s Second'%(end_time-start_time)) #输出运行时间

    W_TRAINED = sess.run(W,feed_dict={x:batch_xs, y:batch_ys})
    np.save("W_TRAINED.npy", W_TRAINED)
    b_TRAINED = sess.run(b,feed_dict={x:batch_xs, y:batch_ys})
    np.save("b_TRAINED.npy", b_TRAINED)

    plt.imshow(W_TRAINED)
    plt.show()
