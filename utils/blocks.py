import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

class SameBlock2d(tf.keras.Model):
  def __init__(self, num_features):
    super(SameBlock2d, self).__init__()
    self.padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
    self.conv = layers.Conv2D(num_features, (3, 3), strides=1, padding=[1, 1], use_bias=False)
    self.batch_norm = layers.BatchNormalization()
  
  def call(self, input_layer):
    block = self.conv(input_layer)
    block = self.batch_norm(block)
    block = layers.ReLU()(block)

    return block

class ResBlock2d(tf.keras.Model):
  def __init__(self, num_features):
    super(ResBlock2d, self).__init__()
    self.conv1 = layers.Conv2D(num_features, (3, 3), strides=1, padding=[1, 1], use_bias=False)
    self.conv2 = layers.Conv2D(num_features, (3, 3), strides=1, padding=[1, 1], use_bias=False)
    self.batch_norm1 = layers.BatchNormalization()
    self.batch_norm2 = layers.BatchNormalization()
  
  def call(self, input_layer):
    block = self.batch_norm1(input_layer)
    block = layers.ReLU()(block)
    block = self.conv1(block)
    block = self.batch_norm2(block)
    block = layers.ReLU()(block)
    block = self.conv2(block)

    block += input_layer

    return block

class DownBlock2d(tf.keras.Model):
  def __init__(self, num_features, norm, pool):
    super(DownBlock2d, self).__init__()
    self.norm = norm
    self.pool = pool
    self.conv = layers.Conv2D(num_features, (4, 4), strides=1, padding="valid")
    self.instance_norm = tfa.layers.InstanceNormalization(axis=3)
  
  def call(self, input_layer):
    block = self.conv(input_layer)
    if self.norm:
      block = self.instance_norm(block)
    block = layers.ReLU()(block)
    if self.pool:
      block = layers.AveragePooling2D()(block)

    return block

class DownBlock(tf.keras.Model):
  def __init__(self, num_features):
    super(DownBlock, self).__init__()
    self.padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
    self.conv = layers.Conv2D(num_features, (3, 3), strides=1, padding=self.padding, use_bias=False)
    self.batch_norm = layers.BatchNormalization()
  
  def call(self, input_layer):
    block = self.conv(input_layer)
    block = self.batch_norm(block)
    block = layers.ReLU()(block)
    block = layers.AveragePooling2D()(block)

    return block

class UpBlock(tf.keras.Model):
  def __init__(self, num_features):
    super(UpBlock, self).__init__()
    self.padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
    self.conv = layers.Conv2D(num_features, (3, 3), strides=1, padding=self.padding, use_bias=False)
    self.batch_norm = layers.BatchNormalization()
  
  def call(self, input_layer):
    block = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(input_layer)
    block = self.conv(block)
    block = self.batch_norm(block)
    block = layers.ReLU()(block)

    return block