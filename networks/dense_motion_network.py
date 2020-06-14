import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

from utils.util import make_coordinate_grid, keypoints_to_gaussian
from utils.models import Hourglass, AntiAliasInterpolation

class DenseMotionNetwork(tf.keras.Model):
  def __init__(self):
    super(DenseMotionNetwork, self).__init__()
    # input shape height x width x 44
    self.num_blocks = 5
    self.num_channels = 3
    self.num_keypoints = 10
    self.scale_factor = 0.25
    self.kp_variance = 0.01
    self.down_features_list = [128, 256, 512, 1024, 1024]
    self.up_features_list = [1024, 512, 256, 128, 64]
    self.padding = [[0, 0], [3, 3], [3, 3], [0, 0]] # pad only height, width

    self.hourglass = Hourglass(self.down_features_list, self.up_features_list, self.num_blocks) # Outputs height x width x 67
    self.mask = layers.Conv2D(self.num_keypoints + 1, (7, 7), strides=1, padding="valid")
    self.occlusion = layers.Conv2D(1, (7, 7), strides=1, padding="valid")
    self.down = AntiAliasInterpolation(self.num_channels, self.scale_factor)
  
  def create_heatmap_representations(self, image_size, kp_driving, kp_source):
    spatial_size = image_size[1:4]
    # shape 256 x 256

    gaussian_driving = keypoints_to_gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
    gaussian_source = keypoints_to_gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
    # shape batch, 10, 256, 256

    heatmap = gaussian_driving - gaussian_source
    # batch x 10 x 256 x 256

    zeros = tf.zeros((heatmap.shape[0], 1, spatial_size[0], spatial_size[1]), dtype=heatmap.dtype)
    # shape batch x 1 x 256 x 256

    heatmap = tf.concat([zeros, heatmap], axis=1)
    # shape batch x 11 x 256 x 256

    heatmap = tf.expand_dims(heatmap, axis=-1)
    # shape batch x 11 x 256 x 256 x 1

    return heatmap
  
  def create_sparse_motions(self, image_size, kp_driving, kp_source):
    batch_size, height, width, _ = image_size
    # Z in equation (4)
    identity_grid = make_coordinate_grid((height, width), type=kp_source['value'].dtype)
    # shape 256 x 256 x 2

    identity_grid = tf.expand_dims(tf.expand_dims(identity_grid, axis=0), axis=0)
    # shape 1 x 1 x 256 x 256 x 2

    # TD<-R in equation (4)
    driving_keypoints = kp_driving['value']
    # shape batch x 10 x 2
    shape = driving_keypoints.shape[:2] + (1, 1, 2)
    driving_keypoints = tf.reshape(driving_keypoints, shape)
    # shape batch, 10, 1, 1, 2

    # Z - TD<-R in equation (4)
    coordinate_grid = identity_grid - driving_keypoints
    # shape batch, 10, 256, 256, 2

    # Using the inverse of d/dp Td <- R ; Equation (5) Jk
    jacobian = tf.linalg.matmul(kp_source['jacobian'], tf.linalg.inv(kp_driving['jacobian']))
    # shape batch x 10 x 2 x 2

    jacobian = tf.expand_dims(tf.expand_dims(jacobian, axis=-3), axis=-3)
    # shape batch x 10 x 1 x 1 x 2 x 2

    jacobian = tf.tile(jacobian, [1, 1, height, width, 1, 1])
    # shape batch x 10 x 256 x 256 x 2 x 2

    # Jk . (Z - TD<-R) in equation (4)
    coordinate_grid = tf.linalg.matmul(jacobian, tf.expand_dims(coordinate_grid, axis=-1))
    # shape batch x 10 x 256 x 256 x 2 x 1

    coordinate_grid = tf.squeeze(coordinate_grid) # remove last axis
    # shape batch x 10 x 256 x 256 x 2

    source_keypoints = kp_source['value']
    # shape batch x 10 x 2    

    shape = source_keypoints.shape[:2] + (1, 1, 2)
    source_keypoints = tf.reshape(source_keypoints, shape)
    # shape batch x 10 x 1 x 1 x 2

    # Ts <- D(z) where source_keypoints is TS<-R and coordinate_grid is Jk . (Z - TD<-R)
    driving_to_source = source_keypoints + coordinate_grid 
    # shape batch x 10 x 256 x 256 x 2
               
    # Adding background feature, background feature is just the identity_grid without motions
    identity_grid = tf.tile(identity_grid, [batch_size, 1, 1, 1, 1])
    # shape batch x 1 x 256 x 256 x 2

    sparse_motions = tf.concat([identity_grid, driving_to_source], axis=1)
    # shape batch x 11 x 256 x 256 x 2 
    # 11 channels since we estimate the taylor aproximation for each keypoint

    return sparse_motions

  def create_deformed_source_image(self, source_image, sparse_motions):
    batch_size, _, height, width = source_image.shape
    # batch x 256 x 256 x 3

    source_repeat = tf.expand_dims(tf.expand_dims(source_image, axis=1))
    # batch x 1 x 256 x 256 x 3
    
    source_repeat = tf.tile(source_repeat, [1, self.num_keypoints + 1, 1, 1, 1])
    # batch x 11 x 256 x 256 x 3

    source_repeat = tf.reshape(source_repeat, [batch_size * (self.num_keypoints + 1), height, width, -1])
    # (batch . 11) x 256 x 256 x 3
    
    sparse_motions = tf.reshape(sparse_motions, [batch_size * (self.num_keypoints + 1), height, width, -1])
    # (batch . 11) x 256 x 256 x 2

    new_max = width - 1
    new_min = 0
    sparse_motions = (new_max - new_min) / (tf.keras.backend.max(sparse_motions) - tf.keras.backend.min(sparse_motions)) * (sparse_motions - tf.keras.backend.max(sparse_motions)) + new_max

    sparse_deformed = tfa.image.resampler(source_repeat, sparse_motions)
    # (batch . 11) x 256 x 256 x 3

    sparse_deformed = tf.reshape(sparse_deformed, [batch_size, (self.num_keypoints + 1), height, width, -1])
    # batch x 11 x 256 x 256 x 3

    return sparse_deformed
  
  def call(self, source_image, kp_driving, kp_source):
    source_image = self.down(source_image)
    image_size = source_image.shape
    batch_size, height, width, _ = image_size
    out_dict = dict()

    heatmap_representation = self.create_heatmap_representations(image_size, kp_driving, kp_source)
    # shape batch x 11 x 256 x 256 x 1
    sparse_motion = self.create_sparse_motions(image_size, kp_driving, kp_source) # keypoint k of d to s
    # shape batch x 11 x 256 x 256 x 2
    warped_images = self.create_deformed_source_image(source_image, sparse_motion)
    # shape batch x 11 x 256 x 256 x 3

    # Debug/print
    out_dict['warped_images'] = warped_images # sparse_deformed

    input = tf.concat([heatmap_representation, warped_images], axis=-1)
    # shape batch x 22 x 256 x 256 x 3

    input = tf.permute(input, [0, 2, 3, 1, 4])
    # shape batch x 256 x 256 x 22 x 3
    
    input = tf.reshape(input, [batch_size, height, width, -1])
    # shape batch x 256 x 256 x 66

    prediction = self.hourglass(input)
    # batch x height x width x 35

    prediction = tf.pad(prediction, self.padding)

    mask = self.mask(prediction)
    # batch x height x width x 11

    mask = tf.keras.activations.softmax(mask)
    # Along the last axis. Thus, we don't repeat values along axes (each keypoint only appears in one channel)
    # batch x height x width x 11

    # Debug/print
    out_dict['mask'] = mask

    mask = tf.expand_dims(mask, axis=-1)
    # batch x height x width x 11 x 1

    sparse_motion = tf.transpose(sparse_motion, [0, 2, 3, 1, 4])
    # batch x 256 x 256 x 11 x 2

    deformation = (sparse_motion * mask)
    deformation = tf.keras.backend.sum(sparse_motion, axis=3) 
    # batch x 256 x 256 x 2

    out_dict['dense_optical_flow'] = deformation # deformation

    occlusion_map = tf.keras.activations.sigmoid(self.occlusion(prediction))
    # shape batch x 256 x 256 x 1

    out_dict['occlusion_map'] = occlusion_map

    return out_dict