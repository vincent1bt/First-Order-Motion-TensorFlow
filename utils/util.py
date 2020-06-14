import tensorflow as tf
import tensorflow_addons as tfa

def detach_keypoint(keypoint):
  return { key: tf.stop_gradient(value) for key, value in keypoint.items() }

def interpolate_tensor(tensor_input, final_shape):
  original_shape = tensor_input.shape[1]

  if final_shape > original_shape:
    return interpolate_increase_size(tensor_input, final_shape)
  else:
    return interpolate_reduce_size(tensor_input, final_shape)

def interpolate_increase_size(tensor_input, final_shape):
  original_shape = tensor_input.shape[1]
  difference = final_shape - original_shape
  border = difference / 2

  padding = [[0, 0], [int(border), int(border)], [int(border), int(border)], [0, 0]]

  width = final_shape
  height = final_shape
  x = tf.linspace(border, width - (border + 1), width)
  yy = tf.tile(tf.reshape(x, (-1, 1)), [1, width])
  xx = tf.tile(tf.reshape(x, (1, -1)), [height, 1])

  grid = tf.concat([tf.expand_dims(xx, axis=2), tf.expand_dims(yy, axis=2)], axis=2)

  output = tfa.image.resampler(tf.pad(tensor_input, padding), tf.expand_dims(grid, axis=0))

  return output

def interpolate_reduce_size(tensor_input, final_shape):
  original_shape = tensor_input.shape[1]
  width = original_shape
  height = original_shape
  x = tf.linspace(0.0, width - 1, final_shape)

  yy = tf.tile(tf.reshape(x, (-1, 1)), [1, final_shape])
  xx = tf.tile(tf.reshape(x, (1, -1)), [final_shape, 1])

  grid = tf.concat([tf.expand_dims(xx, axis=2), tf.expand_dims(yy, axis=2)], axis=2)

  output = tfa.image.resampler(tensor_input, tf.expand_dims(grid, axis=0))

  return output

def make_coordinate_grid(spatial_size, type):
  height, width = spatial_size
  x = tf.keras.backend.arange(width, dtype=type)
  y = tf.keras.backend.arange(height, dtype=type)

  x = (2 * (x / (width - 1)) - 1)
  y = (2 * (y / (height - 1)) - 1)
  
  yy = tf.tile(tf.reshape(y, (-1, 1)), [1, width])
  xx = tf.tile(tf.reshape(x, (1, -1)), [height, 1])

  meshed = tf.concat([tf.expand_dims(xx, axis=2), tf.expand_dims(yy, axis=2)], axis=2)
  # shape 256, 256, 2

  return meshed

def keypoints_to_gaussian(keypoints, spatial_size, kp_variance):
  # TD<-R or TS<-R in equation (6)
  mean = keypoints["value"]
  # shape batch, 10, 2

  # Z in equation (6)
  coordinate_grid = make_coordinate_grid(spatial_size, mean.dtype)
  # shape height x width x 2
  
  coordinate_grid = tf.expand_dims(tf.expand_dims(coordinate_grid, axis=0), axis=0)
  # 1 x 1 x height x width x 2

  repeats = mean.shape[:2] + (1, 1, 1)
  # batch x 10 x 1 x 1 x 1
  coordinate_grid = tf.tile(coordinate_grid, multiples=repeats)
  # batch x 10 x height x width x 2

  # Preprocess kp shape
  shape = mean.shape[:2] + (1, 1, 2)
  # batch x 10 x 1 x 1 x 2

  mean = tf.reshape(mean, shape)
  # batch x 10 x 1 x 1 x 2

  mean_sub = (mean - coordinate_grid)
  # batch x 10 x 1 x 1 x 2 -
  # batch x 10 x height x width x 2

  out = tf.exp(-0.5 * tf.keras.backend.sum(mean_sub ** 2, axis=-1) / kp_variance)
  # batch x 10 x height x width

  return out