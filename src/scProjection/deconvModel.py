#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

## Basics
import sys, os, glob, gzip, time, traceback

## Verbosity of tensorflow output. Filters: (1 INFO) (2 WARNINGS) (3 ERRORS)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

## Math
import numpy as np

## tensorflow imports
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import logging
import numpy as np
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from sklearn.model_selection import StratifiedShuffleSplit

from . import utils
from . import vaeTrain

class deconvModel(object):
    def __init__(self, component_data, component_label, mixture_data, save_label=None):
        ## Required parameters
        self.component_data  = component_data
        self.component_label = component_label
        self.mixture_data    = mixture_data
        if save_label is None:
            self.save_label = self.component_label
        else:
            self.save_label = save_label

        ## Deconvolution
        self.celltypes = np.unique(self.component_label)

        ## Log model state and save locations
        self.results_dir = None
        self.flags       = None

        ## train, test split
        self.train_index = None
        self.test_index  = None

        ## Initialize results class
        self.deconvResults = utils.deconvResult(celltypeNames = self.celltypes)

    def deconvolve( self,
                    ## Masking
                    marker_gene_mask = None,
                    hvg_mask         = None,
                    ## Learning rates (Careful changing these)
                    component_learning_rate = 1e-3,
                    proportion_learning_rate = 1e-3,
                    combined_learning_rate  = 1e-4,
                    decay_lr = False,
                    ## Network architecture
                    num_dr = 50,
                    num_latent_dims = 32,
                    num_layers = 3,
                    hidden_unit_2power = 9,
                    batch_norm_layers = False,
                    decoder_var = 'per_sample',
                    latent_x = False,
                    seed = 1234,
                    ## Early stopping
                    early_stopping = True,
                    early_min_step = 100,
                    max_patience = 100,
                    ## VAE training parameters
                    max_steps_component  = 5000,
                    batch_size_component = 100,
                    corr_check           = False,
                    ## Combined training parameters
                    deconvolution       = True,
                    batch_correction    = True,
                    max_steps_combined = 10000,
                    batch_size_mixture   = 100,
                    max_steps_proportion = 1000,
                    ## KL term
                    KL_weight            = 1.0,
                    marker_weight        = 1,
                    ## MCMC
                    num_monte_carlo      = 15,
                    ## L2 Reg term
                    tied_model_l2        = True,
                    ## Training Setup
                    training_method = 'train',
                    num_folds       = 1,
                    heldout_percent = 0.33,
                    ## Hardware params
                    cuda_device = 0,
                    ## Logging params
                    log_step        = 5000,
                    print_step      = 100,
                    log_results     = True,
                    save_results    = False,
                    log_samples     = True,
                    m_phate_logging = False,
                    ## Path params
                    logdir     = './tmp/',
                    save_to_object = True,
                    save_to_disk   = False,
                    model_name = 'default' ):

        ## Try to run deconv, if fail, be sure to clean up tensorflow properly.
        try:
            ## Setup annotation and masks
            component_labels,  \
            marker_gene_masks, \
            hvg_masks,         \
            max_steps_components = utils.setup_annotations_masks(self.component_label,
                                                                 marker_gene_mask,
                                                                 hvg_mask,
                                                                 max_steps_component,
                                                                 self.celltypes)
            ## Tensorflows variant of argparse
            FLAGS = flags.FLAGS

            ## Result flags
            flags.DEFINE_string('logdir', os.path.abspath(logdir), 'Save directory.')
            flags.DEFINE_string('model', model_name, 'Name of data to use for saving.')

            ## Saving flags
            flags.DEFINE_boolean('log_samples', str(log_samples), 'To log samples from each VAE')
            flags.DEFINE_boolean('m_phate_logging', str(m_phate_logging), 'Logging for m-phate viz')
            flags.DEFINE_boolean('log_results', str(log_results), 'To log detailed results and model files')
            flags.DEFINE_boolean('save_results', str(save_results), 'To save detailed results and model files')
            flags.DEFINE_boolean('save_to_object', str(save_to_object), 'Purified data is saved to results object')
            flags.DEFINE_boolean('save_to_disk', str(save_to_disk), 'Purified data is saved out to disk')

            ## Reporting
            flags.DEFINE_integer('log_step', log_step, 'When to output model summaries.')
            flags.DEFINE_integer('print_step', print_step, 'When to print out sumamries.')
            flags.DEFINE_boolean('monitor_bulk_mse', False, "Should bulk mse be reported during scVAE training")

            ## Prior
            flags.DEFINE_integer('num_dr', num_dr, 'Number of dimensions prior to network.')

            ## Component model architecture flags
            flags.DEFINE_integer('latent_dims', str(num_latent_dims), 'dim of latent Z')
            flags.DEFINE_float('KL_weight', KL_weight, 'Weight on KL term of elbo.')
            flags.DEFINE_boolean('KL_warmup', True, 'Slowly increase KL_weight during training.')
            flags.DEFINE_integer('num_monte_carlo', num_monte_carlo, 'number of samples to estimate likelihood')
            flags.DEFINE_string('decoder_variance', decoder_var, 'Structure of decoder distribution') ## "per_gene", "per_sample", 'default (unit variance)'
            flags.DEFINE_boolean('latent_x', latent_x, 'Should the purified data be treated as latent or not')

            ## Early stopping flags
            flags.DEFINE_boolean('early_stopping', early_stopping, 'Should early stopping be performed.')
            flags.DEFINE_integer('early_min_step', early_min_step, 'How many steps before early stopping procedure starts.')
            flags.DEFINE_integer('max_patience', max_patience, 'How long to wait before early stopping.')

            flags.DEFINE_float('component_learning_rate', component_learning_rate, 'Initial learning rate for VAE training.')
            flags.DEFINE_float('proportion_learning_rate', proportion_learning_rate, "Initial learning rate for proportion est.")
            flags.DEFINE_float('combined_learning_rate', combined_learning_rate, 'Initial learning rate for VAE correction.')
            flags.DEFINE_float('min_learning_rate', 1e-8, 'Minimum learning rate.')
            flags.DEFINE_boolean('decay_lr', str(decay_lr), 'Should learning rate be decayed')
            flags.DEFINE_float('decay_rate', 0.3, 'How fast to decay learning rate.')
            flags.DEFINE_integer('decay_step', 1000, 'Decay interval')

            flags.DEFINE_integer('max_steps_component', str(max(max_steps_components.values())), 'maximum number of traning iterations')
            flags.DEFINE_integer('max_steps_combined', str(max_steps_combined), 'maximum number of traning iterations')
            flags.DEFINE_integer('batch_size_component', str(batch_size_component), 'size of minibatch')
            flags.DEFINE_integer('batch_size_mixture', str(batch_size_mixture), 'size of minibatch')
            flags.DEFINE_boolean('batch_norm', str(batch_norm_layers), 'To include batch_norm layers in model')

            flags.DEFINE_boolean('kl_analytic', True, 'Built in or manual KL computation, current version of tensorflow probability has a bug. Set to False.')
            flags.DEFINE_integer('seed', str(seed), 'random seed for reproducability')

            ## Data specific flags
            flags.DEFINE_boolean('deconvolution', deconvolution, 'Should deconvolution be performed')
            flags.DEFINE_boolean('rec_project', False, 'Will pc space be used as input')
            flags.DEFINE_boolean('L2_norm_data', False, 'Will perform gene-wise l2 normalization.')

            ## Maskings
            flags.DEFINE_string('component', None, 'Subset to just one component (for debugging)')
            flags.DEFINE_string('component_remove', None, 'Remove specific component from reference data.')

            flags.DEFINE_boolean('tpm_softmax', False, 'Should TPM data be rescaled instead of reconstruction to actual values')
            flags.DEFINE_integer('tpm_scale_factor', 10000, 'Scale factor for TPM')

            ## Component model training
            flags.DEFINE_boolean('train_component', True, 'Should component network weights be (re)trained.')
            flags.DEFINE_integer('num_layers', str(num_layers), "Number of encoder/decoder NN layer.")
            flags.DEFINE_integer('hidden_unit_2power', str(hidden_unit_2power), "Starting number of hidden units in the first layer.")
            flags.DEFINE_boolean('tied_model_l2', tied_model_l2, 'L2 regularization between prior and current model weights')

            ## Combined model flags
            flags.DEFINE_integer('proportion_steps', str(max_steps_proportion), 'How long to train combined model before updating VAE(s)')
            flags.DEFINE_boolean('mixture_softmax', True, 'Should softmax be used to compute mixture probabilities.')
            flags.DEFINE_boolean('combined_corr_check', corr_check, 'Should pearson correlation of mixture and reconstructed mixture be recorded? (Slow)')
            flags.DEFINE_boolean('component_corr_check', corr_check, 'Should pearson correlation of mixture and reconstructed mixture be recorded? (Slow)')

            ## Combined loss weights
            flags.DEFINE_boolean('batch_correction', batch_correction, 'Should combined network update components')
            flags.DEFINE_integer('marker_weight', marker_weight, 'Initial learning rate for VAE training.')

            ## regularizer options (UNUSED)
            flags.DEFINE_float('l1_reg_weight', 1e-5, 'Weight of l1 regularizer for combined model.')
            flags.DEFINE_string('input_weight_reg', 'None', 'Should the input gene specific weights have a regularizer')   ## Sparse adjustment
            flags.DEFINE_string('output_weight_reg', 'None', 'Should the output geme specific weights have a regularizer') ## Sparse adjustment
            flags.DEFINE_boolean('mix_weight_reg', False, 'Should the mixture weights have a regularizer') ## Mixture weights

            ## validation setup
            flags.DEFINE_string('training_method', training_method, 'How many, if any, folds to perform CV')
            flags.DEFINE_integer('kfold_validation', str(num_folds), "How many, if any, folds to perform CV")
            flags.DEFINE_float('heldout_percent', str(heldout_percent), "How much data to holdout for testing")

            ## HARDWARE
            flags.DEFINE_string('cuda_device', str(cuda_device), 'Select the GPU for this job')

            ## Remove some logging
            logging.getLogger('tensorflow').setLevel(logging.FATAL)

            ## Hardware configurations
            config = utils.define_hardware(FLAGS)

            ## Setup saving directory
            FLAGS = utils.initialize_model_dir(FLAGS)

            ## Write out all run options for reproducability
            utils.write_flags(FLAGS)

            ## Adjust data based on training mode
            if FLAGS.training_method == "train":
                deconvResults = vaeTrain.runVAE(resultsObj = self.deconvResults,
                                                 FLAGS=FLAGS,
                                                 config=config,
                                                 component_data=self.component_data,
                                                 component_labels=component_labels,
                                                 save_labels=self.save_label,
                                                 mixture_data=self.mixture_data,
                                                 marker_gene_masks=marker_gene_masks,
                                                 hvg_masks=hvg_masks,
                                                 max_steps_components=max_steps_components,
                                                 component_reconstruction_data=None,
                                                 mixture_reconstruction_data=self.mixture_data,
                                                 component_valid_data=None,
                                                 component_valid_labels=None,
                                                 mixture_valid_data=None,
                                                 mixture_valid_labels=None,
                                                 loadings=None)
            elif FLAGS.training_method == "validate":
                ## Setup kfold iterator object
                split_iter = StratifiedShuffleSplit(n_splits=2, test_size=FLAGS.heldout_percent, random_state=FLAGS.seed)
                for train_index, test_index in split_iter.split(self.component_data, component_labels):
                    deconvResults = vaeTrain.runVAE(resultsObj = self.deconvResults,
                                                     FLAGS=FLAGS,
                                                     config=config,
                                                     component_data=self.component_data[train_index,:],
                                                     component_labels=component_labels[train_index],
                                                     save_labels=self.save_label,
                                                     mixture_data=self.mixture_data,
                                                     marker_gene_masks=marker_gene_masks,
                                                     hvg_masks=hvg_masks,
                                                     max_steps_components=max_steps_components,
                                                     component_reconstruction_data=None,
                                                     mixture_reconstruction_data=self.mixture_data,
                                                     component_valid_data=self.component_data[test_index,:],
                                                     component_valid_labels=component_labels[test_index],
                                                     mixture_valid_data=None,
                                                     mixture_valid_labels=None,
                                                     loadings=None)
                    self.train_index = train_index
                    self.test_index  = test_index
            else:
                print("Mode not recognized, leaving data as is.")

            ## Final logging
            self.results_dir = FLAGS.logdir
            self.flags = tf.app.flags.FLAGS.flag_values_dict()
        except:
            print("An error occured: ")
            traceback.print_exc()
        finally:
            ## Clean up
            FLAGS.remove_flag_values(FLAGS.flag_values_dict())

    def check_args(self):
        print("test")
