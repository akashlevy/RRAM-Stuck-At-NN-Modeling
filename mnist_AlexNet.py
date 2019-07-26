import tensorflow as tf
import numpy as np
import time
import os
import sys
import copy
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from my_module.data_input.dataset import DATA_SET
# tf.enable_eager_execution()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
class data_set(object):
  pass

def quantize(var, lower=-1e-1, upper=1e-1, level_num=2):    #stochastic method
  x = var 
  lenth = upper - lower
  space = lenth / (level_num - 1)
  
  r_shift = np.fix((x - lower) / space) * space + lower   # for "+": lower, for "-": upper
  l_shift = np.fix((x - upper) / space) * space + upper   # for "+": upper, for "-": lower
  probability = (x - r_shift)/space
  random_num = np.random.random(x.shape)
  return np.clip(np.where(
    probability > random_num, \
    l_shift, \
    r_shift \
    ), lower, upper)

def add_collection(var=[], collection=[]):
  for j in var:
    for i in collection:
      tf.add_to_collection(i, j)
  return 0

# function to operate dataset
def _parse_function(example_proto, aug=False):
  features = {"image_raw": tf.FixedLenFeature((), tf.string),
              "label": tf.FixedLenFeature((), tf.int64), 
              "height": tf.FixedLenFeature((), tf.int64), 
              "width" : tf.FixedLenFeature((), tf.int64), 
              "depth" : tf.FixedLenFeature((), tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  if aug==True:
    return tf.image.resize_images(
            tf.image.random_flip_left_right(
            tf.image.random_flip_up_down(
            tf.image.random_brightness(
            tf.image.per_image_standardization(
            tf.reshape(tf.io.decode_raw(parsed_features["image_raw"], out_type=tf.uint8), [28,28,1])), 
            max_delta=0.3))), [224,224]), tf.cast(tf.one_hot(parsed_features["label"], 10), dtype=tf.float32)
  else:
    return tf.image.resize_images(
            tf.image.per_image_standardization(
            tf.reshape(tf.io.decode_raw(parsed_features["image_raw"], out_type=tf.uint8), [28,28,1])), 
            [224,224]), tf.cast(tf.one_hot(parsed_features["label"], 10), dtype=tf.float32)

  # return tf.io.decode_raw(parsed_features["image"], output_type=tf.int8), parsed_features["label"]
  #       # parsed_features["height"], parsed_features["width"], parsed_features["depth"]

def batch_norm(x, output_type, name, *args):
  with tf.variable_scope(name):
    if output_type=="flat":
      gamma = tf.get_variable('gamma', initializer=np.array([1], dtype=np.float32))
      betta = tf.get_variable('betta', initializer=np.array([0], dtype=np.float32))
    else:
      gamma = tf.get_variable('gamma', initializer=np.ones(x.shape.as_list()[3], dtype=np.float32))
      betta = tf.get_variable('betta', initializer=np.zeros(x.shape.as_list()[3], dtype=np.float32))
    try:
      x_, mean, variance = tf.nn.fused_batch_norm(
          x, mean=None, variance=None, offset=betta, scale=gamma, epsilon=1e-5, name='BN')
    except ValueError:
      x = tf.reshape(x, [-1, 1, x.shape.as_list()[1], 1])
      x_, mean, variance = tf.nn.fused_batch_norm(
          x, mean=None, variance=None, offset=betta, scale=gamma, epsilon=1e-5, name='BN')
      x_ = tf.reshape(x_, [-1, x_.shape.as_list()[1]*x_.shape.as_list()[2]])
    finally:
      add_collection([gamma, betta], [args])
      return x_

def dataset_input(path=[], prefetch=10000, aug=False):
  TFRecord = tf.data.TFRecordDataset(path).shuffle(5000, reshuffle_each_iteration=True)
  TFRecord = TFRecord.map(lambda t: _parse_function(t, aug), 
            num_parallel_calls=2).prefetch(prefetch)
  # iterator = TFRecord.batch(batch_size).make_initializable_iterator()
  return TFRecord

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def inference(x):      # x is 224 by 224
  cnn = x
  with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE) as scope:
    filter_num = 48 
    kernel_init = np.random.normal(loc=0, scale=0.01, size=[11, 11, cnn.shape.as_list()[3], filter_num])
    bias_init = np.random.normal(loc=0, scale=0.01, size=[filter_num])
    kernel = tf.get_variable('kernel', initializer=kernel_init.astype(np.float32), collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    bias   = tf.get_variable('bias'  , initializer=bias_init.astype(np.float32), collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
    kernel.mas = tf.get_variable('kernel_mas', initializer=kernel.initial_value)
    bias.mas   = tf.get_variable('bias_mas'  , initializer=bias.initial_value)
    variable_summaries(kernel, "kernel")
    variable_summaries(bias, "bias")
    variable_summaries(kernel.mas, "kernel_mas")
    variable_summaries(bias.mas, "bias_mas")
    add_collection([kernel, bias], ["quan"])
    cnn = tf.nn.conv2d(cnn, kernel, strides=[1, 4, 4, 1], padding='SAME')
    cnn = tf.nn.bias_add(cnn, bias)

  with tf.variable_scope("BN1"):
    gamma = tf.get_variable('gamma', initializer=np.ones(cnn.shape.as_list()[3], dtype=np.float32))
    betta = tf.get_variable('betta', initializer=np.zeros(cnn.shape.as_list()[3], dtype=np.float32))
    cnn, mean, variance = tf.nn.fused_batch_norm(
          cnn, mean=None, variance=None, offset=betta, scale=gamma, epsilon=1e-5, name='BN')
    cnn = tf.nn.relu(cnn)

  with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE) as scope:
    filter_num = 128
    kernel_init = np.random.normal(loc=0, scale=0.01, size=[5, 5, cnn.shape.as_list()[3], filter_num])
    bias_init = np.random.normal(loc=0, scale=0.01, size=[filter_num])
    kernel = tf.get_variable('kernel', initializer=kernel_init.astype(np.float32), collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    bias   = tf.get_variable('bias'  , initializer=bias_init.astype(np.float32), collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
    kernel.mas = tf.get_variable('kernel_mas', initializer=kernel.initial_value)
    bias.mas   = tf.get_variable('bias_mas'  , initializer=bias.initial_value)  
    variable_summaries(kernel, "kernel")
    variable_summaries(bias, "bias")
    variable_summaries(kernel.mas, "kernel_mas")
    variable_summaries(bias.mas, "bias_mas")
    add_collection([kernel, bias], ["quan"])
    cnn = tf.nn.conv2d(cnn, kernel, strides=[1, 1, 1, 1], padding='SAME')
    cnn = tf.nn.bias_add(cnn, bias)
  
  cnn = tf.nn.pool(cnn, [2, 2], 'MAX', 'VALID', strides=[2, 2])

  with tf.variable_scope("BN2"):
    gamma = tf.get_variable('gamma', initializer=np.ones(cnn.shape.as_list()[3], dtype=np.float32))
    betta = tf.get_variable('betta', initializer=np.zeros(cnn.shape.as_list()[3], dtype=np.float32))
    cnn, mean, variance = tf.nn.fused_batch_norm(
          cnn, mean=None, variance=None, offset=betta, scale=gamma, epsilon=1e-5, name='BN')
    cnn = tf.nn.relu(cnn)

  with tf.variable_scope("conv3", reuse=tf.AUTO_REUSE) as scope:
    filter_num = 192
    kernel_init = np.random.normal(loc=0, scale=0.01, size=[3, 3, cnn.shape.as_list()[3], filter_num])
    bias_init = np.random.normal(loc=0, scale=0.01, size=[filter_num])
    kernel = tf.get_variable('kernel', initializer=kernel_init.astype(np.float32), collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    bias   = tf.get_variable('bias'  , initializer=bias_init.astype(np.float32), collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
    kernel.mas = tf.get_variable('kernel_mas', initializer=kernel.initial_value)
    bias.mas   = tf.get_variable('bias_mas'  , initializer=bias.initial_value)
    variable_summaries(kernel, "kernel")
    variable_summaries(bias, "bias")
    variable_summaries(kernel.mas, "kernel_mas")
    variable_summaries(bias.mas, "bias_mas")
    add_collection([kernel, bias], ["quan"])
    cnn = tf.nn.conv2d(cnn, kernel, strides=[1, 1, 1, 1], padding='SAME')
    cnn = tf.nn.bias_add(cnn, bias)

  cnn = tf.nn.pool(cnn, [2, 2], 'MAX', 'VALID', strides=[2, 2])

  with tf.variable_scope("BN3"):
    gamma = tf.get_variable('gamma', initializer=np.ones(cnn.shape.as_list()[3], dtype=np.float32))
    betta = tf.get_variable('betta', initializer=np.zeros(cnn.shape.as_list()[3], dtype=np.float32))
    cnn, mean, variance = tf.nn.fused_batch_norm(
          cnn, mean=None, variance=None, offset=betta, scale=gamma, epsilon=1e-5, name='BN')
    cnn = tf.nn.relu(cnn)

  with tf.variable_scope("conv4", reuse=tf.AUTO_REUSE) as scope:
    filter_num = 192
    kernel_init = np.random.normal(loc=0, scale=0.01, size=[3, 3, cnn.shape.as_list()[3], filter_num])
    bias_init = np.random.normal(loc=0, scale=0.01, size=[filter_num])
    kernel = tf.get_variable('kernel', initializer=kernel_init.astype(np.float32), collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    bias   = tf.get_variable('bias'  , initializer=bias_init.astype(np.float32), collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
    kernel.mas = tf.get_variable('kernel_mas', initializer=kernel.initial_value)
    bias.mas   = tf.get_variable('bias_mas'  , initializer=bias.initial_value)
    variable_summaries(kernel, "kernel")
    variable_summaries(bias, "bias")
    variable_summaries(kernel.mas, "kernel_mas")
    variable_summaries(bias.mas, "bias_mas")
    add_collection([kernel, bias], ["quan"])
    cnn = tf.nn.conv2d(cnn, kernel, strides=[1, 1, 1, 1], padding='SAME')
    cnn = tf.nn.bias_add(cnn, bias)

  with tf.variable_scope("BN4"):
    gamma = tf.get_variable('gamma', initializer=np.ones(cnn.shape.as_list()[3], dtype=np.float32))
    betta = tf.get_variable('betta', initializer=np.zeros(cnn.shape.as_list()[3], dtype=np.float32))
    cnn, mean, variance = tf.nn.fused_batch_norm(
          cnn, mean=None, variance=None, offset=betta, scale=gamma, epsilon=1e-5, name='BN')
    cnn = tf.nn.relu(cnn)

  with tf.variable_scope("conv5", reuse=tf.AUTO_REUSE) as scope:
    filter_num = 128
    kernel_init = np.random.normal(loc=0, scale=0.01, size=[3, 3, cnn.shape.as_list()[3], filter_num])
    bias_init = np.random.normal(loc=0, scale=0.01, size=[filter_num])
    kernel = tf.get_variable('kernel', initializer=kernel_init.astype(np.float32), collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    bias   = tf.get_variable('bias'  , initializer=bias_init.astype(np.float32), collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
    kernel.mas = tf.get_variable('kernel_mas', initializer=kernel.initial_value)
    bias.mas   = tf.get_variable('bias_mas'  , initializer=bias.initial_value)
    variable_summaries(kernel, "kernel")
    variable_summaries(bias, "bias")
    variable_summaries(kernel.mas, "kernel_mas")
    variable_summaries(bias.mas, "bias_mas")
    add_collection([kernel, bias], ["quan"])
    cnn = tf.nn.conv2d(cnn, kernel, strides=[1, 1, 1, 1], padding='SAME')
    cnn = tf.nn.bias_add(cnn, bias)

  cnn = tf.nn.pool(cnn, [2, 2], 'MAX', 'VALID', strides=[2, 2])

  with tf.variable_scope("BN5"):
    gamma = tf.get_variable('gamma', initializer=np.ones(cnn.shape.as_list()[3], dtype=np.float32))
    betta = tf.get_variable('betta', initializer=np.zeros(cnn.shape.as_list()[3], dtype=np.float32))
    cnn, mean, variance = tf.nn.fused_batch_norm(
          cnn, mean=None, variance=None, offset=betta, scale=gamma, epsilon=1e-5, name='BN')
    cnn = tf.nn.relu(cnn)

  with tf.variable_scope('FC1', reuse=tf.AUTO_REUSE) as scope:
    filter_num = 2048
    cnn = tf.reshape(cnn, 
      [-1, cnn.shape.as_list()[1]*cnn.shape.as_list()[2]*cnn.shape.as_list()[3]])
    weight_init = np.random.normal(loc=0, scale=0.01, size=[cnn.shape.as_list()[1], filter_num])
    # weight_init = np.random.uniform(LOWER, UPPER, [cnn.shape.as_list()[1], filter_num])
    bias_init = np.random.normal(loc=0, scale=0.01, size=[filter_num])
    # bias_init = np.random.uniform(LOWER, UPPER, [cnn.shape.as_list()[1], filter_num])
    weight = tf.get_variable('weight', initializer=weight_init.astype(np.float32), collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    bias   = tf.get_variable('bias'  , initializer=bias_init.astype(np.float32), collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
    weight.mas = tf.get_variable('weight_mas', initializer=weight.initial_value)
    bias.mas   = tf.get_variable('bias_mas'  , initializer=bias.initial_value)
    variable_summaries(weight, "weight")
    variable_summaries(bias, "bias")
    variable_summaries(weight.mas, "weight_mas")
    variable_summaries(bias.mas, "bias_mas")
    add_collection([weight, bias], ["quan"])
    cnn = tf.matmul(cnn, weight) + bias

  with tf.variable_scope("BN6"):
    gamma = tf.get_variable('gamma', initializer=np.array([1], dtype=np.float32))
    betta = tf.get_variable('betta', initializer=np.array([0], dtype=np.float32))
    cnn = tf.reshape(cnn, [-1, 1, cnn.shape.as_list()[1], 1])
    cnn, mean, variance = tf.nn.fused_batch_norm(
    cnn, mean=None, variance=None, offset=betta, scale=gamma, epsilon=1e-5, name='BN')
    cnn = tf.reshape(cnn, [-1, cnn.shape.as_list()[1]*cnn.shape.as_list()[2]])
    cnn = tf.nn.relu(cnn)

  with tf.variable_scope('FC2', reuse=tf.AUTO_REUSE) as scope:
    filter_num = 2048
    # weight_init = np.random.uniform(LOWER, UPPER, [cnn.shape.as_list()[1], filter_num])
    weight_init = np.random.normal(loc=0, scale=0.01, size=[cnn.shape.as_list()[1], filter_num])
    # bias_init = np.random.uniform(LOWER, UPPER, [cnn.shape.as_list()[1], filter_num]), 
    bias_init = np.random.normal(loc=0, scale=0.01, size=[filter_num])
    weight = tf.get_variable('weight', initializer=weight_init.astype(np.float32), collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    bias   = tf.get_variable('bias'  , initializer=bias_init.astype(np.float32), collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
    weight.mas = tf.get_variable('weight_mas', initializer=weight.initial_value)
    bias.mas   = tf.get_variable('bias_mas'  , initializer=bias.initial_value)
    variable_summaries(weight, "weight")
    variable_summaries(bias, "bias")
    variable_summaries(weight.mas, "weight_mas")
    variable_summaries(bias.mas, "bias_mas")
    add_collection([weight, bias], ["quan"])
    cnn = tf.matmul(cnn, weight) + bias

  with tf.variable_scope("BN7"):
    gamma = tf.get_variable('gamma', initializer=np.array([1], dtype=np.float32))
    betta = tf.get_variable('betta', initializer=np.array([0], dtype=np.float32))
    cnn = tf.reshape(cnn, [-1, 1, cnn.shape.as_list()[1], 1])
    cnn, mean, variance = tf.nn.fused_batch_norm(
    cnn, mean=None, variance=None, offset=betta, scale=gamma, epsilon=1e-5, name='BN')
    cnn = tf.reshape(cnn, [-1, cnn.shape.as_list()[1]*cnn.shape.as_list()[2]])
    cnn = tf.nn.relu(cnn)

  with tf.variable_scope('output_layer', reuse=tf.AUTO_REUSE) as scope:
    filter_num = 10
    # weight_init = np.random.uniform(LOWER, UPPER, [cnn.shape.as_list()[1], filter_num])
    weight_init = np.random.normal(loc=0, scale=0.01, size=[cnn.shape.as_list()[1], filter_num])
    # bias_init = np.random.uniform(LOWER, UPPER, [cnn.shape.as_list()[1], filter_num]), 
    bias_init = np.random.normal(loc=0, scale=0.01, size=[filter_num])
    weight = tf.get_variable('weight', initializer=weight_init.astype(np.float32), collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    bias   = tf.get_variable('bias'  , initializer=bias_init.astype(np.float32), collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
    weight.mas = tf.get_variable('weight_mas', initializer=weight.initial_value)
    bias.mas   = tf.get_variable('bias_mas'  , initializer=bias.initial_value)
    variable_summaries(weight, "weight")
    variable_summaries(bias, "bias")
    variable_summaries(weight.mas, "weight_mas")
    variable_summaries(bias.mas, "bias_mas")
    add_collection([weight, bias], ["quan"])
    cnn = tf.matmul(cnn, weight) + bias
  return cnn

def training(level_num, description):
  BATCH_SIZE = 500
  LOWER = -1e0
  UPPER = 1e0
  noise = 1e-1
  LEVEL_NUM = level_num
  SPACE = (UPPER - LOWER)/LEVEL_NUM
  work_dir = os.getcwd() + '/'
  description = description
  save_file_name = work_dir + 'result/' + description + '.txt'
  # saver_path = work_dir + description + '_param/' + description + ".ckpt"
  for folder in ['result/', description + '_tmp/', description + '_param/']:
    try: 
      os.makedirs(work_dir + folder)
    except OSError:
      if not os.path.isdir(work_dir + folder):
        raise

  train_data_path = ["mnist_data_TFRecord/train.tfrecords"]
  valid_data_path = ["mnist_data_TFRecord/validation.tfrecords"]

  train = data_set()
  handle = tf.placeholder(dtype=tf.string)
  train_dataset = dataset_input(train_data_path, 2000, aug=True).batch(BATCH_SIZE).repeat()
  valid_dataset = dataset_input(valid_data_path, 2000, aug=False).batch(BATCH_SIZE)
  train_iterator = train_dataset.make_one_shot_iterator()
  valid_iterator = valid_dataset.make_initializable_iterator()
  iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
  train.X, train.y = iterator.get_next()
  tf.summary.image('images', train.X, 10)
  cnn = batch_norm(train.X, "same", "input_BN", "BN")
  y_out = batch_norm(inference(cnn), "flat", "output_BN", "BN")
  
  # with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
  #   kernel = tf.get_variable('kernel')
  #   kernel_transpose = tf.transpose(kernel, perm=[3,0,1,2])
  #   tf.summary.image("input_kernel", kernel_transpose, max_outputs=10)
  
  l2_loss = sum([tf.nn.l2_loss(x) for x in tf.get_collection("quan")])
  # for var in tf.get_collection("quan"):
  #   variable_summaries(tf.abs(var - tf.clip_by_value(var.mas, LOWER, UPPER)), "quantization_error")
  loss = tf.square(train.y - y_out)
  loss_origin = tf.reduce_mean(loss, name='loss')
  loss = loss_origin + 0*1e-4 * l2_loss
  predition = tf.nn.softmax(y_out)
  correct_predition = tf.equal(tf.argmax(predition, 1), tf.argmax(train.y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
  global_step = tf.get_variable("global_step", initializer=0, trainable=False)
  train_OP = tf.train.AdamOptimizer(1e-2)
  grad = train_OP.compute_gradients(loss, 
  [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if var not in tf.get_collection("quan")])
  try:
    train_step = train_OP.apply_gradients(grad)
  except ValueError:
    pass
  train_OP_quan = tf.train.AdamOptimizer(1e-1 * SPACE)     # LR ~ 0.1 * space
  grad_quan = train_OP_quan.compute_gradients(loss, tf.get_collection("quan"))
  grad_quan_modified = [[grad_var[0], grad_var[1].mas] for grad_var in grad_quan]
  train_step_quan = train_OP_quan.apply_gradients(grad_quan_modified, global_step=global_step)
  init = tf.global_variables_initializer()
  with tf.name_scope("loss"):
    tf.summary.scalar("loss", loss_origin)
    
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.8
  with tf.Session(config=config) as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(description + '/train/', sess.graph)
    # for var in tf.get_collection("quan"):
    #   # var.load(var.mas.eval(sess), sess)
    #   var.load(quantize(var.mas.eval(sess), LOWER, UPPER, LEVEL_NUM), sess)
    
    train_handle = sess.run(train_iterator.string_handle())
    valid_handle = sess.run(valid_iterator.string_handle())

    def load_quan_save(var):
      error_rate = np.random.random(var.mas.shape) < 1.56e-2 * LEVEL_NUM / (LEVEL_NUM - 1)
      error_value = LOWER + quantize((UPPER - LOWER)*np.random.random(var.mas.shape), LOWER, UPPER, LEVEL_NUM)
      quantized_value = quantize(var.mas.eval(sess), LOWER, UPPER, LEVEL_NUM)
      var.load(np.where(error_rate, error_value, quantized_value), sess)
      
    while True:
      time_start = time.time()
      for _ in range(10):
        _= sess.run([train_step_quan], feed_dict={handle: train_handle})
        # for i in range(1):
        #   _ = sess.run([train_step_quan], feed_dict={handle: train_handle})
        for var in tf.get_collection("quan"):
          thread = threading.Thread(target=load_quan_save, args=(var,))
          thread.start()
          thread.join()
          # var.load(var.mas.eval(sess), sess)
          # var.load(quantize(var.mas.eval(sess), LOWER, UPPER, LEVEL_NUM), sess)
      _= sess.run([train_step], feed_dict={handle: train_handle})
      step = sess.run(global_step)
      if step % 20 == 0:
        summary, = sess.run([merged], feed_dict={handle: train_handle})
        train_writer.add_summary(summary, step)

      sess.run(valid_iterator.initializer)
      valid_loss, train_loss, valid_acc, counter = 0, 0, 0, 0
      while True:
        try:
          valid_loss_, valid_acc_ = sess.run([loss_origin, accuracy], feed_dict={handle: valid_handle})
          train_loss_,  = sess.run([loss_origin], feed_dict={handle: train_handle})
          valid_loss += valid_loss_
          train_loss += train_loss_
          valid_acc += valid_acc_
          counter += 1
        except tf.errors.OutOfRangeError:
          break
      time_spent = time.time() - time_start
      valid_loss /= counter
      train_loss /= counter
      valid_acc /= counter
      print("---------------------------------")
      print("took:       %.3fs"%time_spent)
      print("step:       ", step)
      print("train loss: ", train_loss)
      print("valid loss: ", valid_loss)
      print("accuracy:   ", valid_acc)
      try:
        with open(save_file_name, "r+") as foo:
          pass
        with open(save_file_name, "a+") as foo:
          foo.write(str(step)+'\t'+str(train_loss)+'\t'+str(valid_loss)+'\t'+str(valid_acc)+'\t'+'\n')
      except IOError:
        with open(save_file_name, "a+") as foo:
          foo.write("step\t"+"train_loss\t"+"valid_loss\t"+"valid_accuracy\t"+"\n")
          foo.write(str(step)+'\t'+str(train_loss)+'\t'+str(valid_loss)+'\t'+str(valid_acc)+'\t'+'\n')

      if step >= 1e4:
        break


if __name__ == "__main__":
  sample = [4]
  for i in sample:
    description = "mnist_" + str(i) + "levels_"
    training(i,description)

    tf.reset_default_graph()

  print(":D")
