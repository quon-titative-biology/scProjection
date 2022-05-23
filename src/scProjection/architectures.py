from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys, os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import monitored_session
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import training_util

## Simple prior N(0,I)
def vae_prior(ndim,
              scope):
  with tf.name_scope('prior_Z'):
    return tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(ndim),
                                                    scale_diag=tf.ones(ndim),
                                                    allow_nan_stats=False,
                                                    validate_args=True,
                                                    name="normal_prior_Z")

## Approximation: q(z|x)
def vae_encoder(inputs,
                ndim,
                FLAGS,
                scope,
                num_layers=3, ## This number excludes the embedding layer
                hidden_unit_2power=9, ## Specify the maximal number (2^X) hidden units.
                l2_weight=1e-4,
                dropout_rate=0.3,
                batch_norm=True,
                batch_renorm=False,
                is_training=True):
  with tf.name_scope('vae_encoder'):
      inputs = tf.cast(inputs, tf.float32)
      encoder = tf.layers.dense(inputs=inputs,
                                units=pow(2,hidden_unit_2power),
                                activation=tf.nn.relu,
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
                                use_bias=True,
                                bias_initializer=init_ops.zeros_initializer(),
                                name='fc1')
      if batch_norm: encoder = tf.layers.batch_normalization(encoder, training=is_training, renorm=batch_renorm, name='batch_norm_1')
      encoder = tf.layers.dropout(inputs=encoder, rate=dropout_rate, training=is_training, name='drop1')
      ## Add additional layers while accounting for input layer
      for layer in range(1,num_layers):
          encoder = tf.layers.dense(inputs=encoder,
                                    units=pow(2,hidden_unit_2power-layer),
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.glorot_uniform_initializer(),
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
                                    use_bias=True,
                                    bias_initializer=init_ops.zeros_initializer(),
                                    name='fc'+str(layer+1))
          if batch_norm: encoder = tf.layers.batch_normalization(encoder, training=is_training, renorm=batch_renorm, name='batch_norm_'+str(layer+1))
          encoder = tf.layers.dropout(inputs=encoder, rate=dropout_rate, training=is_training, name='drop'+str(layer+1))
      ## Embedding layer has user defined number of hidden units (ndim)
      encoder = tf.layers.dense(inputs=encoder,
                                units=2*ndim,
                                activation=None,
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                kernel_regularizer=None,
                                use_bias=True,
                                bias_initializer=init_ops.zeros_initializer(),
                                name='fc'+str(num_layers+1))
      encoder = tfp.distributions.MultivariateNormalDiag(
            loc=encoder[..., :ndim],
            scale_diag=tf.nn.softplus(encoder[..., ndim:])+1e-8, ## Changed tf.nn.softplas due to numeric issues with large variances (Potentially unused dimensions)
            allow_nan_stats=False,
            validate_args=False,
            name="latent_Z")
  return encoder

## Reconstruction: p(x|z)
def vae_decoder(inputs,
                output_size,
                FLAGS,
                scope,
                l2_weight=1e-4,
                hidden_unit_2power=9,
                num_layers=3,
                dropout_rate=0.3,
                batch_norm=True,
                batch_renorm=False,
                is_training=True):
  with tf.name_scope('vae_decoder'):
      inputs = tf.cast(inputs, tf.float32)
      decoder = tf.layers.dense(inputs=inputs,
                                units=pow(2,(hidden_unit_2power-num_layers)+1),
                                activation=tf.nn.relu,
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
                                use_bias=True,
                                bias_initializer=init_ops.zeros_initializer(),
                                name='fc1')
      if batch_norm: decoder = tf.layers.batch_normalization(decoder, training=is_training, renorm=batch_renorm, name='batch_norm_1')
      decoder = tf.layers.dropout(inputs=decoder, rate=dropout_rate, training=is_training, name='drop1')
      ## Add additional layers while accounting for input layer
      for layer in range(num_layers-2,-1,-1):
          decoder = tf.layers.dense(inputs=decoder,
                                    units=pow(2,hidden_unit_2power-layer),
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.glorot_uniform_initializer(),
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
                                    use_bias=True,
                                    bias_initializer=init_ops.zeros_initializer(),
                                    name='fc'+str(num_layers-layer))
          if batch_norm: decoder = tf.layers.batch_normalization(decoder, training=is_training, renorm=batch_renorm, name='batch_norm_'+str(num_layers-layer))
          decoder = tf.layers.dropout(inputs=decoder, rate=dropout_rate, training=is_training, name='drop'+str(num_layers-layer))
      decoder_mean = tf.layers.dense(inputs=decoder,
                                units=output_size,
                                activation=None,
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                kernel_regularizer=None,
                                use_bias=True,
                                bias_initializer=init_ops.zeros_initializer(),
                                name='fc'+str(num_layers+1))
      if FLAGS.decoder_variance == "per_sample":
          decoder_var = tf.layers.dense(inputs=decoder,
                                    units=1,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.glorot_uniform_initializer(),
                                    kernel_regularizer=None,
                                    use_bias=True,
                                    bias_initializer=init_ops.zeros_initializer(),
                                    name='fc'+str(num_layers+1)+"_var")
          decoder = tfp.distributions.MultivariateNormalDiag(
                loc=decoder_mean,
                scale_diag=(tf.ones(output_size)*tf.nn.softplus(decoder_var)+1e-8),
                allow_nan_stats=False,
                validate_args=False,
                name="reconstructed_cell")
      else:
          decoder_var = tf.ones(1)
          decoder = tfp.distributions.MultivariateNormalDiag(
                loc=decoder_mean,
                scale_diag=tf.ones(output_size),
                allow_nan_stats=False,
                validate_args=False,
                name="reconstructed_cell")
  return decoder #, decoder_var
