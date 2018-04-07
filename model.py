import numpy as np
import librosa
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_PATH = "./data/"

def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

def save_data_to_array(path=DATA_PATH, max_pad_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)
        


def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
    


#save_data_to_array()

X_train, X_test, y_train, y_test = get_train_test()
X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

x = tf.placeholder(tf.float32,shape=[None,20,11,1])
y_true = tf.placeholder(tf.float32,shape=[None,3])
hold_prob = tf.placeholder(tf.float32)

convo_1 = convolutional_layer(x,shape=[2,2,1,32])
#we need to fix max pool layer some dimension issue
#convo_1_pooling = max_pool_2by2(convo_1)  
convo_2_flat = tf.reshape(convo_1,[-1,20*11*32])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
y_pred = normal_full_layer(full_one_dropout,3)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(250):
        print("Training Epoch : " + str(i))
        sess.run(train, feed_dict={x: X_train, y_true: y_train_hot, hold_prob: 0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print(sess.run(acc,feed_dict={x:X_test,y_true:y_test_hot,hold_prob:1.0}))
    
    saver.save(sess,'models/cnn_model.ckpt')



#Prediction on a new voice
from tkinter import filedialog
from tkinter import Tk,Label,Button,Canvas



def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    global path
    filename = filedialog.askopenfile()
    sample = wav2mfcc(filename.name)
    print(filename.name)
    path = "aplay " + filename.name[50:]
    sample_reshaped = sample.reshape(1, 20, 11, 1)
    labels = ["happy", "cat", "bed"]
    with tf.Session() as sess :
        saver.restore(sess,'models/cnn_model.ckpt')
        predict = tf.argmax(y_pred,1)
        pred = sess.run(predict,feed_dict={x:sample_reshaped,y_true: y_train_hot, hold_prob: 1.0})
        ans = labels[pred[0]]
        canvas.create_text(350, 25, text = "The detected word is : " + ans, font=("Purisa", 25)) 
        

root = Tk()
root.title("Welcome to voice predictor") 
root.geometry('800x600')
button2 = Button(text="Browse Audio File", command=browse_button, height = 1, width = 20,font=("Ariel", 15))
button2.pack()
canvas = Canvas(width=700, height=50, bg='green')
canvas.pack()
play = lambda : os.system(path)
button = Button(root, text = 'Play Word Sound', command = play, height = 1, width = 20,font=("Purisa", 15))
button.pack()
root.mainloop()
