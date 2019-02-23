import math
import tensorflow as tf

import lfw_dataset
import matplotlib.pyplot as plt


slim = tf.contrib.slim
batch_size = 2

dataset_dir = 'lfw_tfrecord'
TFrecord_dataset = lfw_dataset



def dataset_input():
    
    dataset = TFrecord_dataset.get_dataset('train', dataset_dir)                                          

                                         
    provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=False,
      common_queue_capacity=2 * batch_size,
      common_queue_min=batch_size,
      )
    
    image, label = provider.get(['image', 'label'])
    
    image.set_shape((250, 250, 3))

   
    images, labels = tf.train.batch(
      tensors=[image, label],
      batch_size=batch_size,
      num_threads=1,
      capacity=5 * batch_size)
    
    return images, labels


images, labels = dataset_input()

sess = tf.Session()
coord = tf.train.Coordinator()
tf.train.start_queue_runners(coord=coord, sess=sess)

for i in range(12):

    my_images, my_labels = sess.run([images, labels])

    print(my_labels[0])
    print(my_labels[1])
    

#plt.imshow(my_images[0]) 
   