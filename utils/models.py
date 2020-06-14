import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

from utils.util import interpolate_tensor, make_coordinate_grid
from utils.blocks import DownBlock, UpBlock

class AntiAliasInterpolation(tf.keras.Model):
  def __init__(self, channels, scale):
    super(AntiAliasInterpolation, self).__init__()
    sigma = (1 / scale - 1) / 2
    kernel_size = 2 * round(sigma * 4) + 1
    self.scale = scale

    kernel_size = [kernel_size, kernel_size]
    sigma = [sigma, sigma]

    kernel = 1
    meshgrids = tf.meshgrid(*[tf.keras.backend.arange(size, dtype='float32') for size in kernel_size], indexing='ij')

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
      mean = (size - 1) / 2
      kernel *= tf.math.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

    kernel = kernel / tf.keras.backend.sum(kernel)

    #[kernel_height, kernel_width, channels, num_kernels))]
    kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
    kernel = tf.tile(kernel, [1, 1, channels, 1])

    # Important since we want to apply the kernel to each channel dimension
    self.conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=1, use_bias=False, padding="same", weights=[kernel])

  def call(self, input):
    if self.scale == 1.0:
      return input

    out = self.conv(input)
    new_size = int(self.scale * input.shape[1])
    out = interpolate_tensor(out, new_size)

    return out

class ImagePyramide(tf.keras.Model):
  def __init__(self, scales):
    super(ImagePyramide, self).__init__()
    self.num_channels = 3
    self.downs = {}
    
    for scale in scales:
      self.downs[str(scale).replace('.', '-')] = AntiAliasInterpolation(self.num_channels, scale)

  def call(self, x):
    out_dict = {}
    
    for scale, down_module in self.downs.items():
      out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)

    return out_dict

class Vgg19(tf.keras.Model):
  def __init__(self):
    layers = ['block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2', 'block5_conv2'] 
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layers]

    self.model = tf.keras.Model([vgg.input], outputs)
  
  def call(self, x):
    x = tf.keras.applications.vgg19.preprocess_input(x)
    return model(x)

class Hourglass(tf.keras.Model):
  def __init__(self, down_features_list, up_features_list, num_blocks=5):
    super(Hourglass, self).__init__()
    encoder_blocks = []

    for i in range(num_blocks):
      num_features = down_features_list[i]
      encoder_blocks.append(DownBlock(num_features))
    
    self.encoder_blocks = encoder_blocks

    decoder_blocks = []

    for i in range(num_blocks):
      num_features = up_features_list[i]
      decoder_blocks.append(UpBlock(num_features))
    
    self.decoder_blocks = decoder_blocks
  
  def call(self, x):
    down_block_list = [x]
    for down_block in self.encoder_blocks:
      down_block_list.append(down_block(down_block_list[-1]))
    
    model = down_block_list.pop()

    for up_block in self.decoder_blocks:
      model = up_block(model)
      skip = down_block_list.pop()
      model = tf.concat([model, skip], axis=-1)
    
    return model

class Transform:
  def __init__(self, batch_size):
    self.sigma_affine = 0.05
    self.sigma_tps = 0.005
    self.points_tps = 5
    noise = tf.random.normal((batch_size, 2, 3), mean=0, stddev=self.sigma_affine)
    # eye returns an identity matrix
    self.theta = noise + tf.expand_dims(tf.eye(2, 3), axis=0)
    # shape batch x 2 x 3
    self.batch_size = batch_size

    self.control_points = make_coordinate_grid((self.points_tps, self.points_tps), type=noise.dtype)
    self.control_points = tf.expand_dims(self.control_points, axis=0)
    # shape 1 x 5 x 5 x 2
    self.control_params = tf.random.normal((batch_size, 1, self.points_tps ** 2), mean=0, stddev=self.sigma_tps)
    # shape batch x 1 x 25
  
  def transform_frame(self, frame):
    grid = make_coordinate_grid(frame.shape[1:3], type=frame.dtype)
    grid = tf.expand_dims(grid, axis=0)
    grid = tf.reshape(grid, [1, frame.shape[1] * frame.shape[2], 2])
    # shape 1 x (height * width) x 2
    grid = self.warp_coordinates(grid)
    # batch x new_size x 2
    grid = tf.reshape(grid, [self.batch_size, frame.shape[1], frame.shape[2], 2])
    # batch x 256 x 256 x 2
    
    new_max = frame.shape[2] - 1
    new_min = 0
    grid = (new_max - new_min) / (tf.keras.backend.max(grid) - tf.keras.backend.min(grid)) * (grid - tf.keras.backend.max(grid)) + new_max

    return tfa.image.resampler(frame, grid)
    # return F.grid_sample(frame, grid, padding_mode="reflection")
  
  def warp_coordinates(self, coordinates):
    theta = tf.cast(self.theta, coordinates.dtype)
    theta = tf.expand_dims(theta, axis=1)
    # shape batch x 1 x 2 x 3

    # coordinates shape can be 
    #     1 x (height * width) x 2
    # batch x num_keypoints x 2
    transformed = tf.linalg.matmul(theta[:, :, :, :2], tf.expand_dims(coordinates, axis=-1)) + theta[:, :, :, 2:]
    # shape batch x (height * width) x 2 x 1 or
    # shape batch x num_keypoints x 2 x 1
    transformed = tf.squeeze(transformed, axis=-1)
    # shape batch x (height * width) x 2

    control_points = tf.cast(self.control_points, coordinates.dtype)
    # shape 1 x 5 x 5 x 2
    control_params = tf.cast(self.control_params, coordinates.dtype)
    # shape batch x 1 x 25

    # coordinates = xi, yi,  control_points = x, y
    distances = tf.reshape(coordinates, [coordinates.shape[0], -1, 1, 2]) - tf.reshape(control_points, [1, 1, -1, 2])
    # shape 1 x new_size x 25 x 2

    distances = tf.keras.backend.sum(tf.abs(distances), axis=-1)
    # shape 1 x new_size x 25

    result = distances ** 2

    result = result * tf.math.log(distances + 1e-6)
    result = result * control_params
    # batch x new_size x 25

    result = tf.keras.backend.sum(result, axis=2)
    # batch x new_size
    result = tf.reshape(result, [self.batch_size, coordinates.shape[1], 1])
    # batch x new_size x 1
    transformed = transformed + result
    # batch x new_size x 2

    return transformed

  def jacobian(self, coordinates, tape):
    new_coordinates = self.warp_coordinates(coordinates)
    x = tf.keras.backend.sum(new_coordinates[..., 0])
    y = tf.keras.backend.sum(new_coordinates[..., 1])

    grad_x = tape.gradient(x, coordinates) 
    grad_y = tape.gradient(y, coordinates)

    return tf.concat([tf.expand_dims(grad_x, axis=-2), tf.expand_dims(grad_y, axis=-2)], axis=-2)