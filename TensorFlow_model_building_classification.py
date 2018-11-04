import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split

def loadImages(path):
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        loadedImages.append(plt.imread(path + image))
    return np.array(loadedImages)

panel = loadImages('./photo_small/panel/') / 255
modern = loadImages('./photo_small/modern/') / 255
photo = np.concatenate((panel, modern), axis=0)
label_first = np.concatenate((np.zeros(20), np.ones(20)), axis=0)
label_second = np.concatenate((np.ones(20), np.zeros(20)), axis=0)
label_almost = np.vstack((label_first, label_second))
label = label_almost.swapaxes(1,0)
X_train, X_test, y_train, y_test = train_test_split(photo, label, test_size=0.1, random_state=42)
X_train_placeholder = tf.placeholder(shape=[None, 200, 150, 3],dtype=tf.float32)
y_train_placeholder = tf.placeholder(shape=[None, 2],dtype=tf.float32)
nn = tf.layers.conv2d(X_train_placeholder, 32, kernel_size=(3, 3), activation='relu')
nn = tf.layers.conv2d(nn, 64, kernel_size=(3, 3), activation='relu')
nn = tf.layers.max_pooling2d(nn, pool_size=(2, 2), strides=2)
nn = tf.layers.dropout(nn, 0.25)
nn = tf.layers.flatten(nn)
nn = tf.layers.dense(nn, 128, activation='relu')
nn = tf.layers.dropout(nn, 0.5)
nn = tf.layers.dense(nn, 2)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train_placeholder, logits=nn))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(40):
        _, loss_val = sess.run([optimizer,cross_entropy], feed_dict={X_train_placeholder: X_train, y_train_placeholder:y_train})
        if i%2 == 0:
            matches = tf.equal(tf.argmax(nn,1),tf.argmax(y_train_placeholder,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print('Currently on step {}'.format(i))
            print('Loss: ', str(loss_val))
            print('Training accuracy is:')
            print(sess.run(acc,feed_dict={X_train_placeholder: X_train, y_train_placeholder: y_train}))
            print('Validation accuracy is:')
            print(sess.run(acc,feed_dict={X_train_placeholder: X_test, y_train_placeholder: y_test}))
            print('\n')



