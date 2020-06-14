import tensorflow as tf
from tensorflow.keras import layers

from utils.util import make_coordinate_grid
from utils.models import Hourglass, AntiAliasInterpolation

# height, width, channels = 3
class KeypointDetector(tf.keras.Model):
  def __init__(self):
    super(KeypointDetector, self).__init__()
    self.scale_factor = 0.25
    self.num_jacobian_maps = 10
    self.num_keypoints = 10
    self.num_channels = 3
    self.down_features_list = [64, 128, 256, 512, 1024]
    self.up_features_list = [512, 256, 128, 64, 32]
    self.num_blocks = 5
    self.temperature = 0.1

    self.predictor = Hourglass(self.down_features_list, self.up_features_list, self.num_blocks) # Outputs height x width x 35
    self.keypoints_map = layers.Conv2D(self.num_keypoints, (7, 7), strides=1, padding='valid')

    # Initialize the weights/bias with identity transformation. localisation network
    weigth_initializer = tf.keras.initializers.zeros()
    bias_initializer = tf.keras.initializers.constant([1, 0, 0, 1] * 10)
    self.jacobian = layers.Conv2D(self.num_keypoints * 4, (7, 7), strides=1, padding='valid', bias_initializer=bias_initializer, kernel_initializer=weigth_initializer)

    self.down = AntiAliasInterpolation(self.num_channels, self.scale_factor)
  
  def get_gaussian_keypoints(self, heatmap):
    # heatmaps are confidence maps
    # compute soft-argmax (get the coords of the maximum values of the heatmap) differentiable
    heatmap = tf.expand_dims(heatmap, -1)
    # shape batch x 250 x 250 x 10 x 1
    grid = make_coordinate_grid(heatmap.shape[1:3], heatmap.dtype)
    # shape 250 x 250 x 2
    grid = tf.expand_dims(grid, axis=2)
    # shape 250 x 250 x 1 x 2
    grid = tf.expand_dims(grid, axis=0)
    # shape 1 x 250 x 250 x 1 x 2

    value = heatmap * grid
    # shape batch x 250 x 250 x 10 x 2
    value = tf.keras.backend.sum(value, axis=[1, 2])
    # shape batch x 10 x 2

    # keypoints are in a range [-1, 1] due to the grid
    kp = {'value': value}

    return kp
  
  def call(self, x):
    model = self.down(x)
    feature_map = self.predictor(model)
    raw_keypoints = self.keypoints_map(feature_map)

    final_shape = raw_keypoints.shape # pytorch 4, 10, 5, 5 tf: 4, 5, 5, 10

    heatmap = tf.keras.activations.softmax(raw_keypoints / self.temperature, axis=[1, 2])
    # temperature increase the values so is easier to compute the soft-argmax
    final_keypoints = self.get_gaussian_keypoints(heatmap)

    jacobian_map = self.jacobian(feature_map)
    # batch x height x width x 40

    jacobian_map = tf.reshape(jacobian_map, [final_shape[0], final_shape[1], final_shape[2], self.num_jacobian_maps, 4])
    # batch x height x width x 10 x 4

    heatmap = tf.expan_dims(heatmap, axis=-1)
    # batch x height x width x 10 x 1

    jacobian = heatmap * jacobian_map # reduce the importance of the places far from the keypoints coords
    # batch x height x width x 10 x 4

    jacobian = tf.reshape(jacobian, [final_shape[0], -1, final_shape[3], 4])
    # batch x (height * width) x 10 x 4

    jacobian = tf.keras.backend.sum(jacobian, axis=1)
    # batch x 10 x 4

    jacobian = tf.reshape(jacobian, [jacobian.shape[0], jacobian.shape[1], 2, 2])
    # batch x 10 x 2 x 2
    # shape batch, 10, 2, 2 where 10: keypoints each with a jacobian of size 2x2

    final_keypoints['jacobian'] = jacobian

    return final_keypoints