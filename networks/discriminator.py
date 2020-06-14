import tensorflow as tf
from tensorflow.keras import layers

from utils.util import keypoints_to_gaussian
from utils.blocks import DownBlock2d

class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.num_channels = 3
    self.num_keypoints = 10
    self.num_blocks = 4
    self.num_features_list = [64, 128, 256, 512] 
    self.kp_variance = 0.01

    encoder_blocks = []

    for i in range(num_blocks):
      num_features = self.num_features_list[i]
      norm = i != 0
      pool = i != num_blocks - 1
      encoder_blocks.append(DownBlock2d(num_features, norm, pool))
    
    self.encoder_blocks = encoder_blocks
    self.conv = layers.Conv2D(1, kernel_size=1)

  def call(self, x, key_points):
    feature_maps = []
    out = x
    # batch x height x width x 3
    heatmap = keypoints_to_gaussian(key_points, x.shape[1:3], self.kp_variance)
    # batch x 10 x height x width
    out = tf.concat([out, tf.permute(heatmap, [0, 2, 3, 1])], axis=-1)
    # batch x height x width x 13

    for block in self.encoder_blocks:
      feature_maps.append(block(out))
      out = feature_maps[-1]
    
    prediction_map = self.conv(out)
    # batch x height x width x 1

    return feature_maps, prediction_map