import tensorflow as tf
import numpy as np
import cv2
import os
#import matplotlib.pyplot as plt

#GENERATING IMAGES
img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

wind_row, wind_col = img.shape[0]/9, img.shape[1]/9

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x+4, y+4, image[y:y + windowSize[1], x:x + windowSize[0]])

def show_window():
    n=1
    for(x,y, window) in sliding_window(img, img.shape[0]/9, (wind_row,wind_col)):
        if window.shape[0] != wind_row or window.shape[1] != wind_col:
            continue
        clone = img.copy()
        cv2.rectangle(clone, (x, y), (x + wind_row, y + wind_col), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        t_img = img[y:y+wind_row-1,x:x+wind_col-3]
        t_img = t_img[int(t_img.shape[1]*(0.08)):int(t_img.shape[1]*(0.95)), int(t_img.shape[0]*(0.08)):int(t_img.shape[0]*(0.93))]
        m='d/'+str(n)+'.jpg'
        cv2.imwrite(m,t_img)
        n=n+1
        cv2.waitKey(1)
show_window()


#PROCESSING
imgs = []
nos = []

for i in range(10):
    path = 'dataset/'+str(i)+'/'
    images = []
    nos.append(len(next(os.walk(path))[2]))
    for j in range (nos[i]):
        images.append(  cv2.resize( cv2.imread(path+str(j+1)+'.jpg') , (28,28))  )

    imgs.append(np.asarray(images)/255.)

# plt.imshow(imgs[1][7])
# plt.show()
#print imgs[1][7].shape

learning_rate = 0.01
training_epochs = 250

n_input = 2352
n_hidden = 25
n_classes = 10

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
weights = {
    'h': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
    'b': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def model(x):
    layer_1 = tf.add(tf.matmul(x,weights['h']) ,biases['b'])
    output_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
    return output_layer

logits = model(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost=0.
        for i in range(10):
            label=np.zeros((1,10))
            label[0][i]=1
            for j in range(nos[i]):
                _,c = sess.run([train_op, loss_op], feed_dict={X: np.reshape(imgs[i][j], [1,2352]), Y: label} )
                avg_cost += c / 148.
        if epoch % 50 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print "Optimization Finished!"

    pred = tf.nn.softmax(logits)

    # TESTING :
    # count = 0
    # for i in range(10):
    #     for j in range(nos[i]):
    #         res = sess.run(pred, feed_dict={X : np.reshape(imgs[i][j], [1,2352])})
    #         print res
    #         r = res[0].tolist()
    #         if r.index(max(r)) == i:
    #             count=count+1
    # print "Accuracy : ", count/148.
    # print sess.run(pred, feed_dict={X : np.reshape(imgs[9][6], [1,2352])})
    # print sess.run(pred, feed_dict={X : np.reshape(imgs[0][17], [1,2352])})
    # print sess.run(pred, feed_dict={X : np.reshape(imgs[8][16], [1,2352])})

    for i in range(1,82):
        t_i = np.asarray( cv2.resize( cv2.imread( 'd/'+str(i)+'.jpg' ) , (28,28)) / 255.)
        res = sess.run(pred, feed_dict={X : np.reshape(t_i, [1,2352])})
        r = res[0].tolist()
        print r.index(max(r)),
        if (i%9 == 0):
            print "\n"
