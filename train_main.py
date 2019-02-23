# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import shutil
import os
#import sys
#import argparse

import numpy as np
import time
import math
#import argparse
#
#
from datasets import dataset_factory
from preprocessing import preprocessing_factory
from nets import nets_factory


slim = tf.contrib.slim


# 数据集
tf.app.flags.DEFINE_integer('train_batch_size', 32, '')
tf.app.flags.DEFINE_integer('eval_batch_size', 102, '')
tf.app.flags.DEFINE_integer('train_image_size', 6149, '')
tf.app.flags.DEFINE_integer('eval_image_size', 1020, '')
tf.app.flags.DEFINE_string('dataset_name', 'flowers', '')
tf.app.flags.DEFINE_string('dataset_dir', 'flower_TFrecord', '')


#网络结构
tf.app.flags.DEFINE_string('model_name', 'mobilenet_v2', '')
tf.app.flags.DEFINE_string('finetune_exclude_scopes', 'MobilenetV2/Logits', '')
tf.app.flags.DEFINE_integer('image_size', 224, '')
tf.app.flags.DEFINE_integer('num_classes', 102, '')


# 训练参数设置

#   保存目录
tf.app.flags.DEFINE_string('finetune_checkpoint', 'checkpoint/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt', '')
tf.app.flags.DEFINE_string('save_model_dir', 'float_flower_train_model', '')
tf.app.flags.DEFINE_string('save_log_dir', 'float_flower_train_log', '')

#   一些参数
tf.app.flags.DEFINE_integer('num_epochs', 50, '')
tf.app.flags.DEFINE_bool('quantize', False, '')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, '')
tf.app.flags.DEFINE_float('init_learning_rate', 1e-3, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 2, '')
tf.app.flags.DEFINE_bool('add_regularization', True, '')
tf.app.flags.DEFINE_string('optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')



FLAGS = tf.app.flags.FLAGS



def TFrecord_dataset_input(is_training):
    
    
    if is_training:
        dataset_split = 'train'
        batch_size = FLAGS.train_batch_size 
    else:
        dataset_split = 'validation'
        batch_size = FLAGS.eval_batch_size 
                    
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, dataset_split, FLAGS.dataset_dir)
                                                  
    provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          shuffle=is_training,
          common_queue_capacity=2 * batch_size,
          common_queue_min=batch_size)
      
    [image, label] = provider.get(['image', 'label'])

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
          FLAGS.model_name, is_training=is_training)

    image = image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)

    images, labels = tf.train.batch(
          [image, label],
          batch_size=batch_size,
          num_threads=4,
          capacity=5 * batch_size)
      
#    labels = slim.one_hot_encoding(labels, FLAGS.num_classes)
    
    return images, labels



def finetune_ckpt_init(sess):
    '''    
    从Imagenet处开始微调，默认加载除了最后一层外的所有变量 
    '''
    if FLAGS.finetune_checkpoint:
        variables_to_restore = []

        exclusions = [scope.strip() for scope in FLAGS.finetune_exclude_scopes.split(',')]

        for var in slim.get_variables_to_restore():
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
            else:
              variables_to_restore.append(var)
                
        slim_init_fn = slim.assign_from_checkpoint_fn(
            FLAGS.finetune_checkpoint,
            variables_to_restore,
            ignore_missing_vars=True)
    
        slim_init_fn(sess)




def _configure_learning_rate(learning_rate, num_samples_per_epoch, global_step):

    decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay / FLAGS.train_batch_size)
    _learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
    
    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(learning_rate,
                                      global_step,
                                      decay_steps,
                                      _learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
        
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.init_learning_rate, name='fixed_learning_rate')




def _configure_optimizer(learning_rate):
    
    if FLAGS.optimizer == 'momentum':    
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=0.9,
            name='Momentum')

    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=0.9,
            momentum=0.9,
            epsilon=1.0)
    
        
    return  optimizer   



def gpu_config():
	"""
	gpu配置
	"""
	# tensorflow gpu配置
	gpuConfig = tf.ConfigProto()
	gpuConfig.allow_soft_placement = False  # 设置为True，当GPU不存在或者程序中出现GPU不能运行的代码时，自动切换到CPU运行
	gpuConfig.gpu_options.allow_growth = True  # 设置为True，程序运行时，会根据程序所需GPU显存情况，分配最小的资源
	gpuConfig.gpu_options.per_process_gpu_memory_fraction = 1  # 程序运行的时，所需的GPU显存资源最大不允许超过rate的设定值
	return gpuConfig



def build_train_model():
  
        
    g = tf.Graph()
    with g.as_default():
        
        with tf.device('/cpu:0'):                        
            images, labels = TFrecord_dataset_input(is_training=True)

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
                                                name = FLAGS.model_name,
                                                num_classes=FLAGS.num_classes,
                                                weight_decay=FLAGS.weight_decay,
                                                is_training=True) 
        logits, end_points = network_fn(images)


        # 设计损失函数  
        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(
                labels = labels, 
                logits = logits,
                )  
        if FLAGS.add_regularization:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)    
            total_loss = tf.add_n([cross_entropy_loss] + regularization_losses)
        else:
            total_loss = cross_entropy_loss
            
      
        if FLAGS.quantize:
            tf.contrib.quantize.create_training_graph()


        #学习率和优化器
        global_step = tf.train.get_or_create_global_step()
        learning_rate = _configure_learning_rate(FLAGS.init_learning_rate, FLAGS.train_image_size, global_step)
        opt = _configure_optimizer(learning_rate)
                    
        train_tensor = slim.learning.create_train_op(
            total_loss,
            optimizer=opt,
            variables_to_train=tf.trainable_variables())

        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('learning_rate', learning_rate)

        
        return g, train_tensor



def train_model(number_of_steps, summary_writer, continue_ckpt_path=None):  
    
    '''
    number_of_steps: 训练的步数
    continue_ckpt_path：若为None，则从Imagenet预训练好的权重开始训练，若不为None,则从该ckpt继续训练 
    '''
    
    g, train_tensor = build_train_model()
    with g.as_default():
        
        with tf.Session(graph=g) as sess:
            
            global_step = tf.train.get_or_create_global_step()
            
            sess.run(tf.global_variables_initializer())
            
            if continue_ckpt_path==None:

                # 加载Imagenet预训练的权重
                finetune_ckpt_init(sess)  
    
                #global_step设置为0      
                sess.run(tf.assign(global_step, 0))
                               
            else:
                #从ckpt处继续训练 
                slim_init_fn = slim.assign_from_checkpoint_fn(
                    continue_ckpt_path,
                    slim.get_variables_to_restore(),
                    ignore_missing_vars=True)            
                slim_init_fn(sess)

                                  
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            merged = tf.summary.merge_all()

            for i in range(number_of_steps):
                
                start_time = time.time()  
                total_loss, steps = sess.run([train_tensor, global_step])
                time_elapsed = time.time() - start_time
                
        #       每隔10步打印信息
                if (steps % 10==0):                        
                    print('global step %d: total_loss = %.4f (%.3f sec/step)' % (steps, total_loss, time_elapsed))


            epoch = steps//math.ceil(FLAGS.train_image_size/FLAGS.train_batch_size)            
                  
            saver=tf.train.Saver(max_to_keep=1)
            
#           即使 max_to_keep=1也会保存多个ckpt，所以保存前先删除之前的。           
            del_dir = os.path.join(os.getcwd(), 'tmp')
            if os.path.exists(del_dir): 
                shutil.rmtree(del_dir)
            saver.save(sess, 'tmp/flower.ckpt', global_step=epoch)            
            ckpt_path = 'tmp/flower.ckpt-' + str(epoch) 

            summary = sess.run(merged)           
            summary_writer.add_summary(summary, epoch)
               
            coord.request_stop()
            coord.join(threads)
            
            return ckpt_path





def build_eval_model(valid_ckpt_path, best_accuracy, best_acc_ckpt, summary_writer):
    '''

    '''
          
    g = tf.Graph()
    with g.as_default():
        
        with tf.device('/cpu:0'):                        
            images, labels = TFrecord_dataset_input(is_training=False)


        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
                                                name = FLAGS.model_name,
                                                num_classes=FLAGS.num_classes,
                                                weight_decay=FLAGS.weight_decay,
                                                is_training=False) 
        logits, end_points = network_fn(images)

      
        if FLAGS.quantize:
            tf.contrib.quantize.create_eval_graph()


        with tf.Session(graph=g) as sess:
            global_step = tf.train.get_or_create_global_step()            

            saver=tf.train.Saver(max_to_keep=1)                   
#            saver.restore(sess, valid_ckpt_path) 
#           加载ckpt 
            slim_init_fn = slim.assign_from_checkpoint_fn(
                valid_ckpt_path,
                slim.get_variables_to_restore(),
                ignore_missing_vars=True)            
            slim_init_fn(sess)
                        
            top_k_op = tf.nn.in_top_k(end_points['Predictions'], labels, 1)
            num_iter = math.ceil(FLAGS.eval_image_size / FLAGS.eval_batch_size)
            true_count = 0
            total_sample_count = num_iter * FLAGS.eval_batch_size
                        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                       
            for i in range(num_iter):               
                predictions = sess.run(top_k_op)
                true_count += np.sum(predictions)

#           当前ckpt的正确率            
            acc = true_count/total_sample_count
            
            steps = sess.run(global_step)
            
            coord.request_stop()
            coord.join(threads)
            
            epoch = steps//math.ceil(FLAGS.train_image_size/FLAGS.train_batch_size)
            print('**********************************************************')
            print('epoch = %d, current epoch accuracy = %f '%(epoch, acc))    
#            print('**********************************************************')

            summary = tf.Summary()
            summary.value.add(tag='accuracy/validation', simple_value=acc)
            summary_writer.add_summary(summary, epoch)


                                
            if acc > best_accuracy:
                
     #           即使 max_to_keep=1也会保存多个ckpt，所以保存前先删除之前的。           
                del_dir = os.path.join(os.getcwd(), FLAGS.save_model_dir)
                if os.path.exists(del_dir): 
                    shutil.rmtree(del_dir)
                               
                saver.save(sess, '%s/flower.ckpt' % FLAGS.save_model_dir, global_step=epoch)
                _best_acc_ckpt = '%s/flower.ckpt-' % FLAGS.save_model_dir + str(epoch)        
                                
                best_accuracy = acc
                best_acc_ckpt = _best_acc_ckpt

        return best_accuracy, best_acc_ckpt        



def main(_):
    
    
    print('Running training')


    summary_writer = tf.summary.FileWriter(FLAGS.save_log_dir, None)
    
#    若 FLAGS.save_model_dir 存在ckpt，则从该处继续训练，(FLAGS.save_model_dir保存的是正确率最高的ckpt)
#   否则从ImaeNet与训练好的ckpt微调    
    if os.path.exists(FLAGS.save_model_dir):    
        best_acc_ckpt = tf.train.latest_checkpoint(FLAGS.save_model_dir)        
        best_accuracy, _ = build_eval_model(best_acc_ckpt, -1, '', summary_writer)        
        tmp_ckpt = best_acc_ckpt
    else:
        best_acc_ckpt = ''
        best_accuracy = 0.0
        tmp_ckpt = ''

    epoch = 0        
    steps_of_1_epoch = math.ceil(FLAGS.train_image_size/FLAGS.train_batch_size)
    
        
    while epoch < FLAGS.num_epochs:
        
        if tmp_ckpt=='':        
            tmp_ckpt=train_model(steps_of_1_epoch, summary_writer, None)
        else:
            tmp_ckpt=train_model(steps_of_1_epoch, summary_writer, tmp_ckpt)

        epoch += 1         
        print('validation running...')
        best_accuracy, best_acc_ckpt = build_eval_model(tmp_ckpt, best_accuracy, best_acc_ckpt, summary_writer)
            

#        print('**********************************************************')        
        print('running epoch : [%d/%d] ' % (epoch, FLAGS.num_epochs))
        print('best_accuracy = %f, best_acc_ckpt = %s'%(best_accuracy, best_acc_ckpt))
        print('**********************************************************')
 
    summary_writer.close()



if __name__ == '__main__':
    tf.app.run()
