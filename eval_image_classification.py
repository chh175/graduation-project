from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import numpy as np
import os

from datasets import dataset_factory
from preprocessing import preprocessing_factory
from nets import nets_factory


slim = tf.contrib.slim


# 数据集种类
num_classes = 102

image_size = 224

#是否量化训练
quantize = False

# 数据集目录
dataset_dir = 'flower_TFrecord'

# 网络名称
model_name = 'mobilenet_v2'

# L2正则化权重衰减
weight_decay = 0.00004

# 测试集数目
eval_imagenet_size = 1020

# 测试集batch_size
eval_batch_size = 102



def TFrecord_dataset_to_eval():
    
  dataset = dataset_factory.get_dataset('flowers', 'test',
                                          dataset_dir)
      
  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=False,
      common_queue_capacity=2 * eval_batch_size,
      common_queue_min=eval_batch_size)
  [image, label] = provider.get(['image', 'label'])

  image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      model_name, is_training=False)

  image = image_preprocessing_fn(image, image_size, image_size)

  images, labels = tf.train.batch(
      [image, label],
      batch_size=eval_batch_size,
      num_threads=4,
      capacity=5 * eval_batch_size)
  
  return images, labels




def eval_model(ckpt_path):
    
    g = tf.Graph()
    with g.as_default():
                
        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
                                                model_name,
                                                num_classes=num_classes,
                                                weight_decay=weight_decay,
                                                is_training=False)

#        with tf.device('/cpu:0'):
        images, labels = TFrecord_dataset_to_eval()
            
        logits, end_points = network_fn(images)
        


        if quantize:
            tf.contrib.quantize.create_eval_graph()
                
        with tf.Session(graph=g) as sess:
            
            saver=tf.train.Saver()       
            saver.restore(sess, ckpt_path)  
                        
            top_k_op = tf.nn.in_top_k(logits, labels, 1)
            num_iter = int(math.ceil(eval_imagenet_size / eval_batch_size))
            true_count = 0
            total_sample_count = num_iter * eval_batch_size
                        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                       
            for i in range(num_iter):               
                predictions = sess.run(top_k_op)
                true_count += np.sum(predictions)

#           当前ckpt的正确率            
            acc = true_count/total_sample_count

            
            coord.request_stop()
            coord.join(threads)
            

        return acc
   
                       

def main(unused_arg):
    ckpt = tf.train.latest_checkpoint('biyesheji\\float_97.45_nol2')
    acc = eval_model(ckpt)
    print('accuracy = %f'%acc)
    
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
