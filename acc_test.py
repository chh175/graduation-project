import numpy as np
import tensorflow as tf
import math

from datasets import dataset_factory
from preprocessing import preprocessing_factory

tf.app.flags.DEFINE_string('dataset_name', 'flowers', '')
tf.app.flags.DEFINE_string('dataset_dir', 'flower_TFrecord', '')
tf.app.flags.DEFINE_string('dataset_split', 'test', '')
tf.app.flags.DEFINE_integer('eval_batch_size', 102, '')
tf.app.flags.DEFINE_string('model_name', 'mobilenet_v2', '')
tf.app.flags.DEFINE_integer('image_size', 224, '')
tf.app.flags.DEFINE_integer('eval_image_size', 1020, '')
tf.app.flags.DEFINE_string('frozen_def_name', 'final_pb/mobilenet_v2_quant.pb', '')


FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim


def TFrecord_dataset_input():
    
    
                    
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split, FLAGS.dataset_dir)
                                                  
    provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          shuffle=False,
          common_queue_capacity=2 * FLAGS.eval_batch_size,
          common_queue_min=FLAGS.eval_batch_size)
      
    [image, label] = provider.get(['image', 'label'])

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
          FLAGS.model_name, is_training=False)

    image = image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)

    images, labels = tf.train.batch(
          [image, label],
          batch_size = FLAGS.eval_batch_size,
          num_threads=4,
          capacity=5 * FLAGS.eval_batch_size)
      
#    labels = slim.one_hot_encoding(labels, FLAGS.num_classes)
    
    return images, labels


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph




g = load_graph(FLAGS.frozen_def_name)
inputs= g.get_tensor_by_name("import/input:0")
outputs= g.get_tensor_by_name("import/output:0")




with tf.Session(graph=g) as sess:
    
    
    images, labels = TFrecord_dataset_input()


    num_iter = int(math.ceil(FLAGS.eval_image_size / FLAGS.eval_batch_size))
    total_sample_count = num_iter * FLAGS.eval_batch_size


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    true_count = 0

    for i in range(num_iter):               
        
        print('batch ', i)
        my_images, my_labels = sess.run([images, labels])    

        top_k_op = tf.nn.in_top_k(outputs, my_labels, 1)

        predictions = sess.run(top_k_op, feed_dict={inputs:my_images})
        true_count += np.sum(predictions)

#           当前ckpt的正确率            
    acc = true_count/total_sample_count
    print('accuracy = %f'%acc)
    
    coord.request_stop()
    coord.join(threads)
