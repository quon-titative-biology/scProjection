"""
Copyright 2018 Quonlab.
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

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

from . import architectures

class modelVAE(object):
  def __init__(self, FLAGS, latent_dims, input_size, output_size, num_samples, reconstruction, loadings, scope, marker_masks=None, hvg_masks=None):
    ## Info
    print("Constructing VAE: "+scope)
    print(" |")

    ## Create component VAE model
    with tf.variable_scope(scope, tf.AUTO_REUSE):
        ## Params for VAE
        self.scope       = scope
        self.latent_dims = latent_dims
        self.input_size  = input_size
        self.output_size = np.count_nonzero(hvg_masks[self.scope]) if hvg_masks is not None else input_size
        self.num_samples = num_samples
        self.batch_size  = np.amin((FLAGS.batch_size_component, self.num_samples))

        ## Maskings
        self.marker_mask = marker_masks[self.scope] if marker_masks is not None else None
        self.non_marker_mask = np.ones(self.output_size)-self.marker_mask if marker_masks is not None else None
        self.hvg_mask    = hvg_masks[self.scope] if hvg_masks is not None else None

        ## Params for early stopping
        self.early_stop  = False
        self.patience    = 0
        self.best_val_loss = 1e20

        ## L2_tied
        self.tied_model_l2 = tf.constant(value=0, dtype=tf.float32)

        ## Architectures (parameterizing dists)
        self.encoder_func = tf.make_template('encoder',
                                             architectures.vae_encoder,
                                             FLAGS=FLAGS,
                                             ndim=self.latent_dims,
                                             num_layers=FLAGS.num_layers,
                                             hidden_unit_2power=FLAGS.hidden_unit_2power,
                                             scope=self.scope,
                                             batch_norm=FLAGS.batch_norm,
                                             create_scope_now_=False)
        self.decoder_func = tf.make_template('decoder',
                                             architectures.vae_decoder,
                                             FLAGS=FLAGS,
                                             output_size=self.output_size,
                                             num_layers=FLAGS.num_layers,
                                             hidden_unit_2power=FLAGS.hidden_unit_2power,
                                             scope=self.scope,
                                             batch_norm=FLAGS.batch_norm,
                                             create_scope_now_=False)

        self.latent_prior = architectures.vae_prior(ndim=self.latent_dims, scope=self.scope)
        self.reconstruction_prior = architectures.vae_prior(ndim=self.output_size, scope=self.scope)

        ## Basics
        self.global_step = tf.train.get_or_create_global_step()
        self.step = tf.Variable(1, name='step', trainable=False, dtype=tf.int32)

        ## Set up inputs. Mini-batch
        with tf.name_scope("component_data"):
            self.index         = tf.range(start=0, limit=self.num_samples, dtype=tf.int32)
            self.data_ph       = tf.placeholder(tf.float32, (None, self.input_size))
            self.rec_data_ph   = tf.placeholder(tf.float32, (None, self.input_size))
            self.train_dataset = tf.data.Dataset.from_tensor_slices((self.data_ph, self.index)).shuffle(self.num_samples).repeat().batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            self.test_dataset  = tf.data.Dataset.from_tensor_slices((self.data_ph, self.index)).batch(tf.shape(self.data_ph, out_type=tf.dtypes.int64)[0]).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            self.iterator      = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                                 self.train_dataset.output_shapes)
            # self.iterator     = self.dataset.make_initializable_iterator()
            self.data_batch, self.data_index = self.iterator.get_next()
            ## Choice of datasets
            self.train_init_op = self.iterator.make_initializer(self.train_dataset)
            self.test_init_op = self.iterator.make_initializer(self.test_dataset)

        ## Holds purified bulk data, helper for combined model
        with tf.name_scope("purified_data"):
            self.pure_data = None

        ## Compute embeddings and logits.
        self.emb = self.encoder_func(inputs=self.data_batch, is_training=True)
        self.latent_sample = self.emb.sample(FLAGS.num_monte_carlo)
        self.rec = self.decoder_func(inputs=self.latent_sample, is_training=True)
        if FLAGS.tpm_softmax is True:
            self.rec_data = tf.nn.softmax(tf.reduce_mean(self.rec.mean(), axis=0), axis=-1) * FLAGS.tpm_scale_factor
        else:
            #self.rec_data = tf.reduce_mean(self.rec.mean(), axis=0)
            self.rec_data = tf.reduce_mean(self.rec.mean(), axis=0)

        ## Set up KL weight warmup
        with tf.name_scope("KL_warmup"):
            if FLAGS.KL_warmup is True: self.KL_weight = tf.cast(tf.minimum(
                                            FLAGS.KL_weight*(tf.cast(self.step, dtype=tf.float32)/(FLAGS.max_steps_component/tf.constant(2.0, dtype=tf.float32))),
                                            FLAGS.KL_weight),
                                            dtype=tf.float32)
            else: self.KL_weight = FLAGS.KL_weight
            tf.compat.v1.summary.scalar("KL_weight", self.KL_weight)
        tf.compat.v1.summary.scalar("training_step", self.step)
        ## Monitor patience counter
        tf.compat.v1.summary.scalar("Patience", self.patience)

        ## Add VAE specific loss
        self.add_ELBO_loss(FLAGS)

        ## Metrics
        self.mse  = tf.losses.mean_squared_error(self.rec_data, self.data_batch, loss_collection="metrics")
        tf.compat.v1.summary.scalar(self.scope+"_component_mse", self.mse)

        ## Correlation for tracking training performance
        if FLAGS.component_corr_check is True:
            self.corr = tfp.stats.correlation(self.rec_data, self.data_batch, sample_axis=-1, event_axis=0)
            self.corr = tf.reduce_mean(tf.linalg.tensor_diag_part(self.corr), axis=0)
            tf.compat.v1.summary.histogram(scope+"_component_correlation", self.corr)

        ## Set up learning rate
        if FLAGS.decay_lr is True:
            with tf.name_scope("learning_rate"):
                self.t_learning_rate = tf.maximum(
                    tf.train.exponential_decay(
                        FLAGS.component_learning_rate,
                        self.step,
                        FLAGS.decay_step,
                        FLAGS.decay_rate),
                    FLAGS.min_learning_rate)
        else:
            self.t_learning_rate = FLAGS.component_learning_rate

        ## Create training operation
        self.update_op=tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope)
        self.create_train_op(self.t_learning_rate, update_ops=self.update_op)

        ## Batch correction loss terms
        if self.marker_mask is not None:
            self.compute_batch_loss(FLAGS)

        ## Record VAE encoder dist. parameters
        # tf.compat.v1.summary.histogram(scope+"_mean", self.emb.mean())
        # tf.compat.v1.summary.histogram(scope+"_variance", self.emb.variance())

        # ## Record VAE decoder dist. parameters
        # tf.compat.v1.summary.histogram(scope+"_mean_decoder", self.rec.mean())
        # tf.compat.v1.summary.histogram(scope+"_variance_decoder", self.rec.variance())

        ## Define weight saving
        #self.record_network_weights()
        #self.tied_weight_regularizer()

        ## Define saving operations
        self.define_saver_ops(FLAGS=FLAGS, is_training=False)
        self.define_inverse_saver_ops(FLAGS=FLAGS, is_training=True)
        self.define_summary_ops()

  def add_logp_loss(self, FLAGS):
    with tf.variable_scope('LogLikelihood'):
        ## `distortion` is the negative log likelihood.
        self.distortion = -self.rec.log_prob(self.data_batch)
        self.avg_distortion = tf.reduce_mean(input_tensor=self.distortion)
        tf.compat.v1.summary.scalar("Reconstruction", self.avg_distortion)

  def add_KL_loss_encoder(self, FLAGS):
    with tf.variable_scope('KL'):
        self.rate = tfp.distributions.kl_divergence(self.emb, self.latent_prior, allow_nan_stats=False)
        self.avg_KL_div = tf.reduce_mean(input_tensor=self.rate)
        tf.compat.v1.summary.scalar("KL_encoder", self.avg_KL_div)

  def add_KL_loss_decoder(self, FLAGS):
    with tf.variable_scope('KL_decoder'):
        self.rate_decoder = tfp.distributions.kl_divergence(self.rec, self.reconstruction_prior, allow_nan_stats=False)
        self.avg_KL_div_decoder = tf.reduce_mean(input_tensor=self.rate_decoder)
        tf.compat.v1.summary.scalar("KL_decoder", self.avg_KL_div_decoder)

  def add_ELBO_loss(self, FLAGS, smoothing=0.0):
    with tf.variable_scope('ELBO'):
        ## Components of elbo
        self.add_logp_loss(FLAGS)
        self.add_KL_loss_encoder(FLAGS)
        ## Per sample loss
        if FLAGS.latent_x is True:
            self.add_KL_loss_decoder(FLAGS)
            elbo_local = -((self.KL_weight * self.rate) + (self.KL_weight * self.rate_decoder) + self.distortion)
        else:
            elbo_local = -((self.KL_weight * self.rate) + self.distortion)
        ## Mean loss for batch
        self.elbo = tf.reduce_mean(elbo_local)
        tf.losses.add_loss(-self.elbo,
                           loss_collection=tf.GraphKeys.LOSSES)
        tf.compat.v1.summary.scalar("elbo", -self.elbo)

  def create_train_op(self, learning_rate, update_ops=None, check_numerics=True, summarize_gradients=False):
    """Create and return training operation."""
    with tf.control_dependencies(update_ops):
        with tf.name_scope('Optimizer'):
            self.l2_reg = tf.losses.get_regularization_loss(scope=self.scope)
            #self.train_loss = tf.add_n(tf.losses.get_losses(loss_collection=tf.GraphKeys.LOSSES, scope=self.scope) + tf.losses.get_regularization_losses(scope=self.scope))
            self.train_loss = tf.add_n(tf.losses.get_losses(loss_collection=tf.GraphKeys.LOSSES, scope=self.scope) + tf.losses.get_regularization_losses(scope=self.scope))
            ## Minimize loss function
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(loss=self.train_loss, global_step=self.step, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope))
            ## Monitor
            #tf.compat.v1.summary.scalar('Learning_Rate_component', learning_rate)
            #tf.compat.v1.summary.scalar('Loss_Total_component', self.train_loss)
            #tf.compat.v1.summary.scalar('Loss_Regularization', self.l2_reg)
            return(self.train_op)

  ## Operation to compute batch correction loss
  def compute_batch_loss(self, FLAGS):
      with tf.variable_scope(self.scope, tf.AUTO_REUSE):
        ## Reconstruction of markers: marker_mask
        marker_measured = tf.multiply(self.data_batch, self.marker_mask)
        marker_reconstruction = tf.multiply(tf.reduce_mean(self.rec.mean(), axis=0), self.marker_mask)
        ## Reweight marker priority using partial observations
        # if FLAGS.marker_weight > 1:
        #     marker_measured = tf.tile(marker_measured, (FLAGS.marker_weight, 1))
        #     marker_reconstruction = tf.tile(marker_reconstruction, (FLAGS.marker_weight, 1))
        self.marker_mse = tf.losses.mean_squared_error(marker_measured,
                                                       marker_reconstruction,
                                                       weights=1.0,
                                                       scope=self.scope)
        ## Reconstruction of non_markers: np.ones - marker_mask
        if np.sum(self.non_marker_mask) > 0:
            non_marker_measured = tf.boolean_mask(self.data_batch, self.non_marker_mask)
            non_marker_reconstruction = tf.multiply(tf.reduce_mean(self.rec.mean(), axis=0), self.non_marker_mask)
            self.non_marker_mse = tf.losses.mean_squared_error(non_marker_measured,
                                                               non_marker_reconstruction,
                                                               weights=1.,
                                                               scope=self.scope)
        else:
            self.non_marker_mse = 0.0

  # ## Operation to compute batch correction loss
  # def compute_batch_loss(self, FLAGS):
  #     with tf.variable_scope(self.scope, tf.AUTO_REUSE):
  #       ## Reconstruction of markers: marker_mask
  #       marker_measured = tf.boolean_mask(self.data_batch, self.marker_mask, axis=1)
  #       marker_reconstruction = tf.boolean_mask(tf.reduce_mean(self.rec.sample(FLAGS.num_monte_carlo)[0], axis=0), self.marker_mask, axis=1)
  #       ## Reweight marker priority using partial observations
  #       if FLAGS.marker_weight > 1:
  #           marker_measured = tf.tile(marker_measured, (FLAGS.marker_weight, 1))
  #           marker_reconstruction = tf.tile(marker_reconstruction, (FLAGS.marker_weight, 1))
  #       self.marker_mse = tf.losses.mean_squared_error(marker_measured,
  #                                                      marker_reconstruction,
  #                                                      weights=1.0,
  #                                                      scope=self.scope)
  #       ## Reconstruction of non_markers: np.ones - marker_mask
  #       if np.sum(self.non_marker_mask) > 0:
  #           non_marker_measured = tf.boolean_mask(self.data_batch, self.non_marker_mask, axis=1)
  #           non_marker_reconstruction = tf.boolean_mask(tf.reduce_mean(self.rec.sample(FLAGS.num_monte_carlo)[0], axis=0), self.non_marker_mask, axis=1)
  #           self.non_marker_mse = tf.losses.mean_squared_error(non_marker_measured,
  #                                                              non_marker_reconstruction,
  #                                                              weights=1.,
  #                                                              scope=self.scope)
  #       else:
  #           self.non_marker_mse = 0.0

  # def record_network_weights(self):
  #     ## Get all weights from current VAE and save just their names
  #     self.weight_names  = [tensor.name for tensor in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope) if tensor.name.endswith('kernel:0')]
  #     self.saved_weights = {}
  #     ## Holds all the ops required to "fill", fixed weight matrices after training VAEs.
  #     for name in self.weight_names:
  #       weight_matrix = tf.get_default_graph().get_tensor_by_name(name)
  #       self.saved_weights[name] = tf.Variable(tf.zeros(weight_matrix.shape), shape=weight_matrix.shape)
  #       #tf.compat.v1.summary.histogram("fixed_"+name[:-2], self.saved_weights[name])

  def assign_network_weights(self, sess, assign=True):
      ## Hardcopy the weight matrices from dense layers
      for name in self.weight_names:
          print("RECORDING: "+name)
          if assign is True:
              self.saved_weights[name].load(sess.run(name))
              np.savetxt('/home/ucdnjj/matrix_files/'+name[:-2].replace("/","_")+'.csv', sess.run(self.saved_weights[name]), delimiter=',')
          else:
              np.savetxt('/home/ucdnjj/matrix_files_end/'+name[:-2].replace("/","_")+'.csv', sess.run(self.saved_weights[name]), delimiter=',')

  # def tied_weight_regularizer(self):
  #     with tf.name_scope('tied_model_l2'):
  #         for name in self.weight_names:
  #             self.tied_model_l2 += tf.nn.l2_loss(tf.subtract(tf.get_default_graph().get_tensor_by_name(name), self.saved_weights[name]))
  #         tf.compat.v1.summary.scalar("tied_model_l2", self.tied_model_l2)

  def define_summary_ops(self):
      ## VAE's summary set
      self.summary_op = tf.summary.merge_all(scope=self.scope)

  ## Defines copies of the network that take in placeholders for final pass of compelete dataset
  def define_saver_ops(self, FLAGS, is_training):
    ## Saver helper function for encoder
    self.fdata_ph = tf.placeholder(tf.float32, (None, self.input_size))
    self.emb_saver = self.encoder_func(inputs=self.fdata_ph, is_training=is_training)
    self.emb_saver_mean = self.emb_saver.mean()
    self.emb_saver_var  = self.emb_saver.variance()
    self.emb_saver_sample = self.emb_saver.sample(FLAGS.num_monte_carlo)[0]

    ## Saver helper function for decoder
    self.emb_ph = tf.placeholder(tf.float32, (None, self.latent_dims))
    self.rec_saver = self.decoder_func(inputs=self.emb_ph, is_training=is_training)
    self.rec_saver_lp = self.rec_saver.log_prob(self.fdata_ph)

    ## Saver helper function for sampled decoder
    self.emb_sampled_ph = tf.placeholder(tf.float32, (None, None, self.latent_dims))
    self.rec = self.decoder_func(inputs=self.emb_sampled_ph, is_training=is_training)
    #self.rec_var  = tf.reduce_mean(self.rec_var, axis=0)
    self.rec_mean = tf.reduce_mean(self.rec.mean(), axis=0)

  def define_inverse_saver_ops(self, FLAGS, is_training=True):
      self.inv_emb_saver = self.encoder_func(inputs=self.fdata_ph, is_training=is_training)
      self.inv_emb_saver_sample = self.inv_emb_saver.sample(FLAGS.num_monte_carlo)[0]
      self.inv_rec = self.decoder_func(inputs=self.emb_sampled_ph, is_training=is_training)
      self.inv_rec_mean = tf.reduce_mean(self.inv_rec.mean(), axis=0)

  def mixture_mse_monitor(self, data, FLAGS, is_training=False):
      emb_comp_temp = sess.run(self.emb_saver_sample, feed_dict={self.fdata_ph: data})
      reconstruction = sess.run(self.rec_mean, feed_dict={self.emb_ph: emb_comp_temp})
      return sess.run(tf.compat.v1.losses.mean_squared_error(data,  reconstruction))

  ## Evaluate endpoint op in batchs of data.
  def save_sampled(self, sess, resultsObj, datatype, scope, FLAGS, dir, nsample=100):
      with tf.variable_scope(self.scope, tf.AUTO_REUSE):
        samples = sess.run(self.latent_prior.sample(nsample))
        rec_comp = sess.run(self.rec_sampled_saver, feed_dict={self.emb_sampled_ph: samples[tf.newaxis,]})
        resultsObj.sample_saver(datatype, scope, samples, rec_comp)
        if FLAGS.save_results is True:
            np.savetxt(FLAGS.logdir + "/"+dir+"_results/reconstructions.csv", rec_comp, delimiter=",")

  ## Save scRNA outputs
  def save_results(self, sess, resultsObj, datatype, mode, stage, FLAGS, data, scope, dir, batch_size=100):
    ## Handle large mixture input
    rec_comp = []; emb_comp  = []; rec_comp_lprob = [];  var_comp  = [];
    with tf.variable_scope(self.scope, tf.AUTO_REUSE):
        for i in range(0, data.shape[0], batch_size):
            data_batch = data[i:i + batch_size]
            emb_comp_temp = sess.run(self.emb_saver_sample, feed_dict={self.fdata_ph: data_batch})
            emb_comp.append(emb_comp_temp)
            rec_comp_lprob.append(sess.run(self.rec_saver_lp, feed_dict={self.emb_ph: emb_comp_temp, self.fdata_ph: data_batch}))
            rec_comp.append(sess.run(self.rec_mean, feed_dict={self.emb_sampled_ph: emb_comp_temp[tf.newaxis,]}))
            #var_comp.append(sess.run(self.rec_var, feed_dict={self.emb_sampled_ph: emb_comp_temp[tf.newaxis,]}))

    ## Combined batched results
    emb_comp = np.concatenate(emb_comp)
    rec_comp_lprob = np.concatenate(rec_comp_lprob)
    rec_comp = np.concatenate(rec_comp)
    #if FLAGS.decoder_variance != 'fixed':
    #    var_comp = np.concatenate(var_comp)
    ##
    # inv_rec_comp = np.concatenate(inv_rec_comp)

    if FLAGS.save_to_object is True:
        resultsObj.update_saver(datatype, mode, stage, scope, rec_comp, emb_comp, rec_comp_lprob, var_comp)

    if FLAGS.save_to_disk is True:
        print("Not setup")
