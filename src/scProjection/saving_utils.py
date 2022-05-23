#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob, gzip, time
from functools import partial
from importlib import import_module

## Math
import numpy as np

## tensorflow imports
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

# def save_mixtures(self, sess, resultsObj, FLAGS, VAE, mixture_input_data, datatype, batch_size=100):
#     batch_size = np.amin((batch_size, mixture_input_data.shape[0]))
#     results=[]
#     ## Loop through bulk in batches
#     for i in range(0, mixture_input_data.shape[0], batch_size):
#         end_itr = np.amin((i + batch_size, mixture_input_data.shape[0]))
#         data_batch = mixture_input_data[i:end_itr,:]
#         ## Compute mixture from weighted components
#         for index, value in enumerate(self.current_celltypes, 0):
#             with tf.variable_scope(value, tf.AUTO_REUSE):
#                 emb      = sess.run(VAE[value].emb_saver_sample, feed_dict={VAE[value].fdata_ph: data_batch})
#                 purified = sess.run(VAE[value].rec_sampled_saver, feed_dict={VAE[value].emb_sampled_ph: emb[tf.newaxis,]})
#             ## Component specific contribution to mixture profile
#             test = sess.run(tf.gather(self.proportions[:,index,tf.newaxis], list(range(i,end_itr)), axis=0))
#             pure_data += sess.run(tf.multiply(purified, tf.gather(self.proportions[:,index,tf.newaxis], list(range(i,end_itr)), axis=0)))
#         results.append(pure_data)
#     resultsObj.mixture_saver(datatype, results)

def save_samples(sess,
                 resultsObj,
                 FLAGS,
                 VAE,
                 component_labels,
                 stage="prebatch"):
    if not os.path.exists(os.path.abspath(FLAGS.logdir + '/' + stage + '/sampled_results')):
        os.makedirs(os.path.abspath(FLAGS.logdir + '/' + stage + '/sampled_results'))
    ## Save component and mixture results
    for scope in np.unique(component_labels):
        print("Saving: "+scope)
        VAE[scope].save_sampled(sess=sess,
                                FLAGS=FLAGS,
                                resultsObj=resultsObj,
                                datatype=stage,
                                scope=scope,
                                dir=stage+"/sampled")

def save_results(sess,
                 resultsObj,
                 FLAGS,
                 VAE,
                 component_labels,
                 component_data,
                 hvg_masks,
                 marker_gene_masks,
                 stage="component",
                 component_valid_data=None,
                 component_valid_labels=None,
                 mixture_input_data=None):
    ## Log a bunch of results!
    if FLAGS.log_results is True:
        ## Create results directories
        if not os.path.exists(os.path.abspath(FLAGS.logdir + '/' + stage + '/component_train_results')):
            os.makedirs(os.path.abspath(FLAGS.logdir + '/' + stage + '/component_train_results'))
        if not os.path.exists(os.path.abspath(FLAGS.logdir + '/' + stage + '/component_valid_results')):
            os.makedirs(os.path.abspath(FLAGS.logdir + '/' + stage + '/component_valid_results'))
        if mixture_input_data is not None:
            if not os.path.exists(os.path.abspath(FLAGS.logdir + '/' + stage + '/mixture_results')):
                os.makedirs(os.path.abspath(FLAGS.logdir + '/' + stage + '/mixture_results'))
        ## Save component and mixture results
        for scope in np.unique(component_labels):
            with tf.variable_scope(scope, tf.AUTO_REUSE):
                print("Saving: "+scope)
                comp_data = component_data[np.where(component_labels == scope)[0],:]
                ## Component vae will only model hvgs specific to cell type
                if mixture_input_data is not None:
                    if hvg_masks is not None:
                        comp_data = comp_data[:,np.nonzero(hvg_masks[scope])[0]]
                        mixture_input_data_scope = mixture_input_data[:,np.nonzero(hvg_masks[scope])[0]]
                    else:
                        mixture_input_data_scope = mixture_input_data

                if comp_data.shape[0] > 0:
                    VAE[scope].save_results(sess=sess,
                                            resultsObj=resultsObj,
                                            datatype="component",
                                            mode="train",
                                            stage=stage,
                                            FLAGS=FLAGS,
                                            data=comp_data,
                                            scope=scope,
                                            dir=stage+"/component_train")
                if component_valid_data is not None and component_valid_data.shape[0] > 0:
                    VAE[scope].save_results(sess=sess,
                                            resultsObj=resultsObj,
                                            datatype="component",
                                            mode="test",
                                            stage=stage,
                                            FLAGS=FLAGS,
                                            data=component_valid_data[np.where(component_valid_labels == scope)[0],:],
                                            scope=scope,
                                            dir=stage+"/component_valid")
                if mixture_input_data is not None:
                    VAE[scope].save_results(sess=sess,
                                            resultsObj=resultsObj,
                                            datatype="purified",
                                            mode="train",
                                            stage=stage,
                                            FLAGS=FLAGS,
                                            data=mixture_input_data_scope,
                                            scope=scope,
                                            dir=stage+"/mixture")
