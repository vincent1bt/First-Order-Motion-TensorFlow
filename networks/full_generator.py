import tensorflow as tf
from tensorflow.keras import layers

from utils import detach_keypoint, Transform
from utils.models import ImagePyramide, Vgg19

class FullGenerator(tf.keras.Model):
  def __init__(self, key_point_detector, generator, discriminator):
    super(FullGenerator, self).__init__()
    self.feature_matching_weights = 10
    self.equivariance_weights = 10
    self.perceptual_weights = [10, 10, 10, 10, 10]
    self.scales = [1, 0.5, 0.25, 0.125]
    self.vgg = Vgg19()
    self.pyramid = ImagePyramide(self.scales)
    self.key_point_detector = key_point_detector
    self.generator = generator
    self.discriminator = discriminator

  def call(self, source_images, driving_images, tape):
    kp_source = self.key_point_detector(source_images)
    kp_driving = self.key_point_detector(driving_images)

    generated = self.generator(source_images, kp_source=kp_source, kp_driving=kp_driving)
    # Debug/print
    generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

    loss_values = {}

    pyramide_real = self.pyramid(driving_images)
    pyramide_generated = self.pyramid(generated['prediction'])

    # Perceptual loss (Loss for gan generator)
    perceptual_loss = 0
    for scale in self.scales:
      x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
      y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

      for i, weight in enumerate(self.perceptual_weights):
        loss = tf.reduce_mean(tf.abs(x_vgg[i] - tf.stop_gradient(y_vgg[i])))
        perceptual_loss += self.perceptual_weights[i] * loss
      loss_values['perceptual'] = perceptual_loss
    
    # Gan loss (only one scale used, the original [1])

    # We detach the keypoints here so we dont compue its gradients and we use it as input images!!!
    discriminator_maps_real, _ = self.discriminator(driving_images, kp=detach_keypoint(kp_driving))
    discriminator_maps_generated, discriminator_pred_map_generated = self.discriminator(generated['prediction'], kp=detach_keypoint(kp_driving))
    
    # LSGAN G Loss
    # Discriminator outputs a pathmap like pix2pix where 1 labels are for real images and 0 labels are for generated images
    # Since we want to fool the discriminator we want our generated images to output 1
    gan_loss = tf.reduce_mean((discriminator_pred_map_generated - 1) ** 2)
    # same as tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.ones_like(discriminator_pred_map_generated), discriminator_pred_map_generated))
    gan_loss += self.loss_weights['generator_gan'] * gan_loss
    loss_values['gen_gan'] = gan_loss
    
    # feature_matching loss
    feature_matching_loss = tf.reduce_mean(tf.abs(discriminator_maps_real - discriminator_maps_generated))
    feature_matching_loss += self.feature_matching_weights * feature_matching_loss

    loss_values['feature_matching'] = feature_matching_loss

    # Equivariance Loss
    batch_size = driving_images.shape[0]
    transform = Transform(batch_size)

    transformed_frame = transform.transform_frame(driving_images)
    # image Y
    # shape batch x height x width x 2

    transformed_keypoints = self.key_point_detector(transformed_frame)
    # Ty <-R

    # Debug/print
    generated['transformed_frame'] = transformed_frame
    # Debug/print
    generated['transformed_kp'] = transformed_keypoints

    keypoints_loss = tf.reduce_mean(tf.abs(kp_driving['value'] - transform.warp_coordinates(transformed_keypoints['value'])))
    loss_values['equivariance_value'] = self.equivariance_weights * keypoints_loss

    # Here we apply the transformation for a second time and then compute the jacobian
    jacobian_transformed = tf.linalg.matmul(transform.jacobian(transformed_keypoints['value'], tape), transformed_keypoints['jacobian'])
    # Equivariance properties

    normed_driving = tf.linalg.inv(kp_driving['jacobian']) #inverse of Tx <-R
    normed_transformed = jacobian_transformed

    jacobian_mul = tf.linalg.matmul(normed_driving, normed_transformed)
    identity_matrix = tf.cast(tf.reshape(tf.eye(2), [1, 1, 2, 2]), jacobian_mul.dtype)
    jacobian_loss = tf.reduce_mean(tf.abs(identity_matrix - jacobian_mul))
    loss_values['equivariance_jacobian'] = self.equivariance_weights * jacobian_loss

    return loss_values, generated