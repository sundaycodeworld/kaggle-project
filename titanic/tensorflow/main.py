# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]

data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Cabin'] = pd.factorize(data['Cabin'])[0]
data.fillna(0, inplace=True)
data['Sex'] = [1 if x=='male' else 0 for x in data['Sex']]
data['p1'] = np.array(data['Pclass']==1).astype(np.int32)
data['p2'] = np.array(data['Pclass']==2).astype(np.int32)
data['p3'] = np.array(data['Pclass']==3).astype(np.int32)
del data['Pclass']
data['e1'] = np.array(data['Embarked']=='S').astype(np.int32)
data['e2'] = np.array(data['Embarked']=='C').astype(np.int32)
data['e3'] = np.array(data['Embarked']=='Q').astype(np.int32)
del data['Embarked']

data_train = data[[ 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'p1', 'p2', 'p3', 'e1', 'e2', 'e3']]
data_target = data['Survived'].values.reshape(len(data), 1)

x = tf.placeholder("float", shape=[None, 12])
y = tf.placeholder("float", shape=[None, 1])

weight = tf.Variable(tf.random_normal([12, 1]))
bias = tf.Variable(tf.random_normal([1]))
output = tf.matmul(x, weight) + bias
pred = tf.cast(tf.sigmoid(output) > 0.5, tf.float32)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
accurary = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

data_test = pd.read_csv('test.csv')
data_test = data_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].mean())
data_test['Cabin'] = pd.factorize(data_test['Cabin'])[0]
data_test.fillna(0, inplace=True)
data_test['Sex'] = [1 if x=='male' else 0 for x in data_test['Sex']]
data_test['p1'] = np.array(data_test['Pclass']==1).astype(np.int32)
data_test['p2'] = np.array(data_test['Pclass']==2).astype(np.int32)
data_test['p3'] = np.array(data_test['Pclass']==3).astype(np.int32)
del data_test['Pclass']
data_test['e1'] = np.array(data_test['Embarked']=='S').astype(np.int32)
data_test['e2'] = np.array(data_test['Embarked']=='C').astype(np.int32)
data_test['e3'] = np.array(data_test['Embarked']=='Q').astype(np.int32)
del data_test['Embarked']

test_label = pd.read_csv('gender_submission.csv')
test_label = np.reshape(test_label['Survived'].values.astype(np.float32), (418,1))

#sess = tf.Session()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
loss_train = []
train_acc = []
test_acc = []

data_train = data_train.values
with tf.device('/gpu:0'):
    for i in range(25000):
        index = np.random.permutation(len(data_target))
        data_train = data_train[index]
        data_target = data_target[index]
        for n in range(len(data_target)//100 + 1):
            batch_xs = data_train[n*100:n*100+100]
            batch_ys = data_target[n*100:n*100+100]
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
            
        if i%1000 == 0:
            loss_temp = sess.run(loss, feed_dict={x:batch_xs, y:batch_ys})
            loss_train.append(loss_temp)
            train_acc_temp = sess.run(accurary, feed_dict={x:batch_xs, y:batch_ys})
            train_acc.append(train_acc_temp)
            test_acc_temp = sess.run(accurary, feed_dict={x:data_test, y:test_label})
            test_acc.append(test_acc_temp)
            print('{:.2f},{:.2%},{:.2%}'.format(loss_temp,train_acc_temp,test_acc_temp))
        
import matplotlib.pyplot as plt

plt.plot(loss_train, 'k-')
plt.title('train loss')
plt.show()

plt.plot(train_acc, 'b-', label='train_acc')
plt.plot(test_acc, 'r--', label='test_acc')
plt.title('train and test accuracy')
plt.legend()
plt.show()






