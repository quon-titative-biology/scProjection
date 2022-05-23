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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import training_util

class deconvModel(object):
  def __init__(self, FLAGS, VAE, datasets, input_size, output_size, num_samples, reconstruction=None, loadings=None, marker_gene_masks=None, hvg_masks=None, scope="prop_model"):
    ## Params for deconvolution model
    self.input_size  = input_size
    self.output_size = output_size
    self.num_samples = num_samples
    self.batch_size  = np.amin((FLAGS.batch_size_mixture, self.num_samples)) ## Batch size should not be larger than dataset size
    ## Marker gene mask
    self.marker_gene_mask = None
    ## Organizing name for cmobined model
    self.scope = scope
    ## Cell types in model
    self.current_celltypes = np.unique(datasets)
    self.num_component = len(self.current_celltypes)
    ##
    self.step = tf.Variable(1, name='prop_step', trainable=False, dtype=tf.int32)

    ## Build combined model
    with tf.variable_scope(self.scope, tf.AUTO_REUSE):
        self.build_dataset();
        self.build_model(FLAGS);
        self.deconvolution(FLAGS, VAE);
        ## MSE between reconstruction and measured mixture
        self.add_prop_loss(self.data_batch, self.mixture_rec)
        ## Create training operation
        self.create_train_op(FLAGS)
        self.create_diagnostics(FLAGS)

  def build_dataset(self):
    with tf.name_scope("mixture_data"):
        ## Mixture indexing
        self.index = tf.range(start=0, limit=self.num_samples, dtype=tf.int32)
        ## Dataset
        self.mix_data_ph = tf.placeholder(tf.float32, (None, self.input_size))
        self.mix_dataset = tf.data.Dataset.from_tensor_slices((self.mix_data_ph, self.index)).shuffle(self.num_samples).repeat().batch(self.batch_size)
        self.mix_iter_data  = self.mix_dataset.make_initializable_iterator()
        self.mix_data_batch, self.data_index  = self.mix_iter_data.get_next()

  def build_model(self, FLAGS):
    with tf.name_scope("mixture_proportions"):
        ## Mixture weights
        self.mixture_weights = tf.get_variable(name="prop_coef",
                                               shape=[self.num_samples, self.num_component],
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer,
                                               regularizer=tf.contrib.layers.l1_regularizer(FLAGS.l1_reg_weight) if (FLAGS.mix_weight_reg is True) else None,
                                               trainable=True)
        ## Normalize mixture weights
        self.proportions = tf.nn.softmax(self.mixture_weights, axis=-1, name="proportions")
    ## Hold results of linear reconstitution of mixture
    self.mixture_rec = tf.zeros([self.batch_size, self.output_size], tf.float32)

  def deconvolution(self, FLAGS, VAE):
    ## Compute mixture from weighted components
    for index, value in enumerate(self.current_celltypes, 0):
        #with tf.name_scope(value):
        with tf.variable_scope(value, tf.AUTO_REUSE):
            data_emb = VAE[value].encoder_func(self.mix_data_batch, is_training=False).sample(FLAGS.num_monte_carlo)
            data_rec = tf.reduce_mean(VAE[value].decoder_func(data_emb, is_training=False).mean(), axis=0)
            ## TPM specific reconstruction
            if FLAGS.tpm_softmax is True:
                data_rec = tf.nn.softmax(data_rec, axis=-1) * FLAGS.tpm_scale_factor
            ## Component specific contribution to mixture profile
            pure_data = tf.multiply(data_rec, tf.gather(self.proportions[:,index,tf.newaxis], self.data_index, axis=0))
        ## Proportion specific reconstructions, could be masked
        if VAE[value].marker_mask is not None:
            self.mixture_rec = tf.add(self.mixture_rec, tf.multiply(pure_data, VAE[value].marker_mask))
            self.data_batch = tf.multiply(self.mix_data_batch, VAE[value].marker_mask)
        else:
            self.mixture_rec = tf.add(self.mixture_rec, pure_data)
            self.data_batch = self.mix_data_batch
        tf.summary.histogram(value + "_weights", self.mixture_weights[:,index])
        tf.summary.histogram(value + "_probs", self.proportions[:,index])

  def add_prop_loss(self, data, rec):
    ## Define decoder reconstruction loss
    self.mse_mixture = tf.losses.mean_squared_error(data,
                                                    rec,
                                                    weights=1.0,
                                                    scope=self.scope)
    tf.summary.scalar("MSE", self.mse_mixture)

  def create_train_op(self, FLAGS):
    """Create and return training operation."""
    with tf.name_scope('Optimizer_prop'):
        ## Set up learning rate
        if FLAGS.decay_lr is True:
            with tf.name_scope("learning_rate_prop"):
                self.prop_learning_rate = tf.maximum(
                    tf.train.exponential_decay(
                        FLAGS.proportion_learning_rate,
                        self.step, ## Account for previous training
                        FLAGS.decay_step*10,
                        FLAGS.decay_rate),
                    FLAGS.min_learning_rate)
        else:
            self.prop_learning_rate = FLAGS.proportion_learning_rate

        ## Collect prop component loss
        self.proportion_loss = self.mse_mixture
        ## Minimize loss function
        optimizer = tf.train.AdamOptimizer(self.prop_learning_rate)
        self.train_proportion_op = optimizer.minimize(loss=self.proportion_loss, global_step=self.step, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope))
        ## Monitor
        tf.summary.scalar('Learning_Rate', self.prop_learning_rate)
        tf.summary.scalar('Loss_Total', self.proportion_loss)
        return(self.train_proportion_op)

  def create_diagnostics(self, FLAGS):
    ## Diagnostic
    if FLAGS.combined_corr_check is True:
      self.corr = tfp.stats.correlation(self.mix_data_batch, self.mixture_rec, sample_axis=-1, event_axis=0)
      self.corr = tf.reduce_mean(tf.linalg.tensor_diag_part(self.corr), axis=0)
      tf.compat.v1.summary.scalar("Mixture_correlation", self.corr)
