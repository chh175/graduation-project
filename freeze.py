import tensorflow as tf
from nets import nets_factory

from tensorflow.python.framework import graph_util


slim = tf.contrib.slim

tf.app.flags.DEFINE_string('model_name', 'mobilenet_v2', '')
tf.app.flags.DEFINE_integer('num_classes', 102, '')
tf.app.flags.DEFINE_integer('image_size', 224, '')
tf.app.flags.DEFINE_bool('quantize', True, '')
tf.app.flags.DEFINE_string('save_model_dir', 'biyesheji/quant_97.05_l2', '')
tf.app.flags.DEFINE_string('output_filename', 'final_pb/mobilenet_v2_quant.pb', '')

FLAGS = tf.app.flags.FLAGS


g = tf.Graph()
with g.as_default():
    inputs = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3], name='input')
        
    network_fn = nets_factory.get_network_fn(
                                            name = FLAGS.model_name,
                                            num_classes=FLAGS.num_classes,
                                            weight_decay=0.0,
                                            is_training=False) 
    logits, end_points = network_fn(inputs)
    
    outputs = tf.identity(end_points['Predictions'], name = 'output')    
        
    if FLAGS.quantize:
        tf.contrib.quantize.create_eval_graph()
    
            
    saver = tf.train.Saver()
    with tf.Session(graph = g) as sess:
        ckpt = tf.train.latest_checkpoint(FLAGS.save_model_dir)
        saver.restore(sess, ckpt)
        
        output_graph_def = graph_util.convert_variables_to_constants(
                sess, g.as_graph_def(), output_node_names=['output']) 
        
        with tf.gfile.FastGFile(FLAGS.output_filename, mode = 'wb') as f:
            f.write(output_graph_def.SerializeToString())