#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob, gzip, time
from functools import partial
from importlib import import_module

## Math
import numpy as np
import scipy as sp

## tensorflow imports
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

## Record evaluation metrics
class modelMetrics(object):
    def __init__(self, mode):
        self.step = {'train': [], 'test': [], 'train-post': []}
        self.loss = {'train': [], 'test': [], 'train-post': []}
        self.mse  = {'train': [], 'test': [], 'train-post': []}
        if mode is "VAE":
            self.log_probability = {'train': [], 'test': [], 'train-post': []}
            self.kl_divergence   = {'train': [], 'test': [], 'train-post': []}

## Record results and evaluation metrics
class deconvResult(object):
    def __init__(self, celltypeNames):
        self.proportions       = {}
        self.weights           = {}
        ## Reconstructions and embeddings
        self.deconv_data = {'component': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}, 'purified': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}}
        self.deconv_emb  = {'component': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}, 'purified': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}}
        self.deconv_logp = {'component': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}, 'purified': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}}
        self.deconv_var  = {'component': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}, 'purified': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}}
        #self.deconv_inv_data = {'component': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}, 'purified': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}}
        ## Reconstructions and embeddings after correction
        self.deconv_data_post = {'component': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}, 'purified': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}}
        self.deconv_emb_post  = {'component': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}, 'purified': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}}
        self.deconv_logp_post = {'component': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}, 'purified': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}}
        self.deconv_var_post  = {'component': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}, 'purified': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}}
        #self.deconv_inv_data_post = {'component': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}, 'purified': {'train': dict.fromkeys(celltypeNames), 'test': dict.fromkeys(celltypeNames)}}
        ## Sampling
        self.deconv_samples_emb = {'prebatch': dict.fromkeys(celltypeNames), 'postbatch': dict.fromkeys(celltypeNames)}
        self.deconv_samples_rec = {'prebatch': dict.fromkeys(celltypeNames), 'postbatch': dict.fromkeys(celltypeNames)}
        ## Mixture reconstruction
        self.deconv_mixtures = {'prebatch': None, 'postbatch': None}
        ## Train/test metrics
        self.component_metrics  = dict([(key, modelMetrics(mode = "VAE")) for key in celltypeNames])
        self.proportion_metrics = modelMetrics(mode = "Mixture")
        self.mixture_metrics   = modelMetrics(mode = "Mixture")
        ## Model details
        self.celltypes        = celltypeNames
        self.results_dir      = None
        self.flags            = None
        ## mphate specifics
        self.m_phate_logging   = dict([(key, []) for key in celltypeNames])

    def update_component_metrics(self, mode, scope, step, loss, mse, logp, kl):
        self.component_metrics[scope].step[mode].append(step)
        self.component_metrics[scope].loss[mode].append(loss)
        self.component_metrics[scope].mse[mode].append(mse)
        self.component_metrics[scope].log_probability[mode].append(logp)
        self.component_metrics[scope].kl_divergence[mode].append(kl)

    def update_mixture_metrics(self, mode, loss, mse):
        self.proportion_metrics.loss[mode].append(loss)
        self.proportion_metrics.mse[mode].append(mse)

    def update_mixture_metrics(self, mode, loss, mse):
        self.mixture_metrics.loss[mode].append(loss)
        self.mixture_metrics.mse[mode].append(mse)

    def update_proportions(self, step, props):
        self.proportions[str(step)] = props
        #self.weights[str(step)]     = weights

    def update_mphate(self, scope, embedding):
        self.m_phate_logging[scope].append(embedding)

    def update_saver(self, datatype, mode, stage, scope, rec_data, embedding, log_prob, rec_var):
        if stage == "component":
            self.deconv_data[datatype][mode][scope] = rec_data
            self.deconv_emb[datatype][mode][scope]  = embedding
            self.deconv_logp[datatype][mode][scope] = log_prob
            self.deconv_var[datatype][mode][scope]  = rec_var
        elif stage == "combined":
            self.deconv_data_post[datatype][mode][scope] = rec_data
            self.deconv_emb_post[datatype][mode][scope]  = embedding
            self.deconv_logp_post[datatype][mode][scope] = log_prob
            self.deconv_var_post[datatype][mode][scope]  = rec_var
        else:
            print("Unknown stage")

    def sample_saver(self, datatype, scope, emb_samples, rec_samples):
        self.deconv_samples_emb[datatype][scope] = emb_samples
        self.deconv_samples_rec[datatype][scope] = rec_samples

    def mixture_saver(self, datatype, mixture_rec):
        self.deconv_mixtures[datatype] = mixture_rec

    def __repr__(self):
        return "Results of <NAME> deconvolution"

def initialize_model_dir(FLAGS):
    print("Running training method: " + FLAGS.training_method)
    ## Ensure save path exists
    FLAGS.logdir = os.path.abspath(FLAGS.logdir)
    ## Setup save path
    FLAGS.logdir = FLAGS.logdir + "/model_" + FLAGS.model
    print(FLAGS.logdir)
    ## Make path
    if not os.path.exists(os.path.abspath(FLAGS.logdir)):
       os.makedirs(os.path.abspath(FLAGS.logdir))
    return FLAGS

## Saves flags to a file
def write_flags(FLAGS):
    flag_dict = tf.app.flags.FLAGS.flag_values_dict()
    with open(FLAGS.logdir + '/run_flags.txt', 'w+') as file_out:
        [file_out.write('{0}\t{1}\n'.format(key, value)) for key, value in flag_dict.items()]

def summary_and_checkpoint(FLAGS, graph):
    ## Tensorboard and model checkpointing
    summary_op = tf.summary.merge_all()
    ## Write summaries
    summary_writer = tf.summary.FileWriter(os.path.abspath(FLAGS.logdir) +  "/train", graph, filename_suffix='-train')
    summary_writer_test = tf.summary.FileWriter(os.path.abspath(FLAGS.logdir) +  "/test", filename_suffix='-test')
    ## Model saving every X steps or Y hours
    saver = tf.train.Saver(max_to_keep=5,  keep_checkpoint_every_n_hours=1.0)
    return summary_op, summary_writer, summary_writer_test, saver

## Setup masking and annotations
def setup_annotations_masks(component_label, marker_gene_mask, hvg_mask, max_steps_component, celltypes):
    ## Convert old format to new dictionary for celltype labels from reference
    if type(component_label) is not dict:
        print("Converting component label list to dictionary")
        component_labels = {'standard_deconv' : component_label}
    else:
        component_labels = component_label

    ## Convert old format to dictionary for marker gene mask which is shared across cell types
    if type(marker_gene_mask) is not dict and marker_gene_mask is not None:
        print("Converting marker mask to dictionary")
        marker_gene_masks = {}
        for scope in celltypes:
            marker_gene_masks.update( {scope : marker_gene_mask} )
    else:
        marker_gene_masks = marker_gene_mask

    ## Convert old format to dictionary for marker gene mask which is shared across cell types
    if type(hvg_mask) is not dict and hvg_mask is not None:
        print("Converting hvg mask to dictionary")
        hvg_masks = {}
        for scope in celltypes:
            hvg_masks.update( {scope : hvg_mask} )
    else:
        hvg_masks = hvg_mask

    ## Convert old format to dictionary for marker gene mask which is shared across cell types
    if type(max_steps_component) is not dict and max_steps_component is not None:
        print("Converting hvg mask to dictionary")
        max_steps_components = {}
        for scope in celltypes:
            max_steps_components.update( {scope : max_steps_component} )
    else:
        max_steps_components = max_steps_component

    return component_labels['standard_deconv'], marker_gene_masks, hvg_masks, max_steps_components

def early_stopping(FLAGS, VAE, scope, corr):
    ## Early stopping if enabled and minimum number of steps has been achieved
    if corr > 0.85:
        VAE[scope].patience = VAE[scope].patience + 1
    elif corr > 0.90:
        VAE[scope].patience = VAE[scope].patience + 10
    else:
        VAE[scope].patience = VAE[scope].patience - 1

    if VAE[scope].patience >= FLAGS.max_patience:
        print("Early stopping triggered for: ", str(scope) + "   Corr: " + str(corr))
        VAE[scope].early_stop = True

def define_hardware(FLAGS):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    config.allow_soft_placement = True
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device
    return config
