import tensorflow as tf

from utils.util import detach_keypoint

class FullDiscriminator(tf.keras.Model):
  def __init__(self, discriminator):
    super(FullDiscriminator, self).__init__()
    self.discriminator = discriminator

  def call(self, x_driving, generated):
    kp_driving = generated['kp_driving']

    loss_values = {}

    _, discriminator_pred_map_real = self.discriminator(x_driving, kp=detach_keypoint(kp_driving))
    _, discriminator_pred_map_generated = self.discriminator(tf.stop_gradient(generated['prediction']), kp=detach_keypoint(kp_driving))
    
    # LSGAN
    discriminator_loss = (1 - discriminator_pred_map_real) ** 2 + discriminator_pred_map_generated ** 2
    # Where discriminator_pred_map_real should output 1's and discriminator_pred_map_generated 0's
    loss_values['disc_gan'] = tf.reduce_mean(discriminator_loss)

    return loss_values