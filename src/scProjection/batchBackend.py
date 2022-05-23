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

class batchModel(object):
    def __init__(self, FLAGS, VAE, datasets, input_size, output_size, num_samples, proportions, mixture_weights, reconstruction=None, loadings=None, VAE_parameters=None, scope="batch_correction"):
        ## Params for deconvolution model
        self.input_size  = input_size
        self.output_size = output_size
        self.num_samples = num_samples
        self.batch_size  = np.amin((FLAGS.batch_size_mixture, self.num_samples)) ## Batch size should not be larger than dataset size
        ## Organizing name for cmobined model
        self.scope = scope
        ## Cell types in model
        self.current_celltypes = np.unique(datasets)
        self.num_component = len(self.current_celltypes)
        ## Normalize mixture weights
        self.mixture_weights = mixture_weights
        self.proportions = proportions

        ## Combined model steps
        self.step = tf.Variable(1, name='batch_step', trainable=False, dtype=tf.int32)

        ## Build combined model
        with tf.variable_scope(self.scope, tf.AUTO_REUSE):
            ## Create parameters and metadata for batch correction
            self.build_dataset();
            self.build_model();
            self.batch_correction(FLAGS, VAE);
            ## MSE between reconstruction and measured mixture during proportion est. or batch correction
            self.add_loss();
            ## Record VAE trainable parameters
            self.trainable_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "bulk_distribution")

            # ## Corr
            # if FLAGS.combined_corr_check is True:
            #     self.corr = tfp.stats.correlation(self.mix_data_batch, self.mixture_rec, sample_axis=-1, event_axis=0)
            #     self.corr = tf.reduce_mean(tf.linalg.tensor_diag_part(self.corr), axis=0)
            #     tf.compat.v1.summary.scalar("Mixture_correlation", self.corr)

            ## Build the optimizer
            self.create_train_op(FLAGS)

    def build_dataset(self):
        with tf.name_scope("mixture_data"):
            ## Mixture indexing
            self.index = tf.range(start=0, limit=self.num_samples, dtype=tf.int32)
            ## Dataset
            self.mix_data_ph = tf.placeholder(tf.float32, (None, self.input_size))
            self.mix_dataset = tf.data.Dataset.from_tensor_slices((self.mix_data_ph, self.index)).shuffle(self.num_samples).repeat().batch(self.batch_size)
            self.mix_iter_data  = self.mix_dataset.make_initializable_iterator()
            self.mix_data_batch, self.data_index  = self.mix_iter_data.get_next()

    def build_model(self):
        ## Hold results of linear reconstitution of mixture
        self.mixture_rec = tf.zeros([self.batch_size, self.output_size], tf.float32)
        ## Batch correction loss
        self.marker_mse     = tf.constant(0., tf.float32)
        self.non_marker_mse = tf.constant(0., tf.float32)
        self.KL_vae         = tf.constant(0., tf.float32)
        self.elbo_vae       = tf.constant(0., tf.float32)
        self.regularization_vae = tf.constant(0., tf.float32)
        self.regularization_tied = tf.constant(0., tf.float32)

    def batch_correction(self, FLAGS, VAE):
        ## Compute mixture from weighted components
        for index, value in enumerate(self.current_celltypes, 0):
            with tf.variable_scope(value, tf.AUTO_REUSE):
                self.data_emb = VAE[value].encoder_func(self.mix_data_batch, is_training=False).sample(FLAGS.num_monte_carlo)
                self.data_dist = VAE[value].decoder_func(self.data_emb, is_training=False)
                self.data_rec = tf.reduce_mean(self.data_dist.mean(), axis=0)

            with tf.name_scope("LinearCombo"):
                ## Component specific contribution to mixture profile
                VAE[value].pure_data = tf.multiply(self.data_rec, tf.gather(self.proportions[:,index,tf.newaxis], self.data_index, axis=0))
                ## Reconstruct mixture celltype-wise
                self.mixture_rec = tf.add(self.mixture_rec, VAE[value].pure_data)
                ## Record metrics
                # if VAE[value].marker_mask is not None:
                #     self.marker_mse      += VAE[value].marker_mse
                #     self.non_marker_mse  += VAE[value].non_marker_mse
                self.KL_vae              += (VAE[value].avg_KL_div * VAE[value].KL_weight)
                self.elbo_vae            += VAE[value].elbo
                self.regularization_vae  += VAE[value].l2_reg
                #self.regularization_tied += VAE[value].tied_model_l2
                tf.compat.v1.summary.scalar('KL_'+ VAE[value].scope, self.KL_vae)
                tf.compat.v1.summary.scalar('ELBO_'+VAE[value].scope, -self.elbo_vae)
                tf.compat.v1.summary.scalar('L2_REG_'+ VAE[value].scope, self.regularization_vae)

            ## Gather VAE variances for bulk variance calculation
            try:
                #self.variances_vae += [tf.ones((10,100))]
                self.trainable_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, VAE[value].scope)
            except AttributeError:
                #self.variances_vae = [tf.ones((10,100))]
                self.trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, VAE[value].scope)
            #tf.summary.histogram(value + "_weights", self.mixture_weights[:,index])
            # tf.summary.histogram(value + "_probs", self.proportions[:,index])

        ## Define bulk distribution for proportion estimation
        with tf.name_scope("bulk_distribution"):
            # if FLAGS.decoder_variance == "per_sample":
            #     merged_vae_vars = tf.concat(self.variances_vae, axis=0)
            #     merged_vae_vars = tf.tile(merged_vae_vars[tf.newaxis,], [self.batch_size,1])
            # else:
            #     merged_vae_vars = tf.stack(self.variances_vae, axis=-1)
            #     merged_vae_vars = tf.tile(merged_vae_vars, [self.batch_size,1])
            # print(merged_vae_vars)
            # print(merged_vae_vars)
            gathered_probs  = tf.gather(self.proportions, self.data_index, axis=0)
            #print(gathered_probs)
            # combined_var_input = tf.concat([gathered_probs, merged_vae_vars], axis=1)
            # print(combined_var_input)
            # combined_var_input.set_shape((self.batch_size, self.num_component+(self.num_component*self.batch_size)))
            # print(combined_var_input)
            self.dist_var = tf.layers.dense(inputs=gathered_probs, #combined_var_input, #tf.tile(tf.reshape(merged_vae_vars, [-1])[tf.newaxis,], [self.batch_size,1])], axis=1),  #[tf.newaxis,]
                                      units=1,
                                      activation=None,
                                      kernel_initializer=tf.glorot_uniform_initializer(),
                                      kernel_regularizer=tf.nn.relu,
                                      use_bias=True,
                                      bias_initializer=init_ops.zeros_initializer(),
                                      name='fc_var')
            self.mixture_rec_dist = tfp.distributions.MultivariateNormalDiag(
                                    loc=self.mixture_rec,
                                    scale_diag=(tf.ones(self.output_size)*tf.nn.softplus(self.dist_var)+1e-8),
                                    allow_nan_stats=False,
                                    validate_args=True,
                                    name="reconstructed_cell_dist")
            ## summaries
            # tf.compat.v1.summary.histogram("prop_bulk_dist_mean", self.mixture_rec_dist.mean())
            # tf.compat.v1.summary.histogram("prop_bulk_dist_var", self.mixture_rec_dist.variance())

    ## Bulk mse loss
    def add_loss(self):
        ## Log probability
        self.batch_loss = -tf.reduce_mean(self.mixture_rec_dist.log_prob(self.mix_data_batch))
        tf.summary.scalar("batch_loss", self.batch_loss)

    ## Create operating to adjust VAEs with fixed proportions
    def create_train_op(self, FLAGS):
        """Create and return training operation."""
        with tf.name_scope('Optimizer_batch'):
            ## Set up learning rate
            if FLAGS.decay_lr is True:
                with tf.name_scope("learning_rate_proportion"):
                    self.batch_learning_rate = tf.maximum(
                        tf.train.exponential_decay(
                            FLAGS.combined_learning_rate,
                            self.step,
                            FLAGS.decay_step,
                            FLAGS.decay_rate),
                        FLAGS.min_learning_rate)
            else:
                self.batch_learning_rate = FLAGS.combined_learning_rate

            ## Minimize loss function
            optimizer = tf.train.AdamOptimizer(self.batch_learning_rate)

            ## Just optimizing proportions based on marker expr. recontruction of bulk
            self.train_batch_loss = self.batch_loss + (-self.elbo_vae) + (self.KL_vae) + tf.reduce_sum(self.regularization_vae)
            if FLAGS.tied_model_l2 is True:
                self.train_batch_loss += tf.reduce_sum(self.regularization_tied)

            ## If markers are provided then we can include marker vs. non-marker mse terms at different magnitudes
            # if FLAGS.marker_weight > 1:
            #     self.train_batch_loss += self.marker_mse + self.non_marker_mse

            ## Batch correction optimization
            self.train_batch_op = optimizer.minimize(loss=self.train_batch_loss, global_step=self.step, var_list=self.trainable_params)

            ## Monitor
            tf.summary.scalar('Loss_Total', self.train_batch_loss)
