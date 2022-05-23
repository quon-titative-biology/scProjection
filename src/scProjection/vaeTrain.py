
#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function ##

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

##
from . import vaeBackend
from . import batchBackend
from . import deconvBackend
from . import utils
from . import saving_utils

def runVAE(resultsObj,
           FLAGS,
           config,
           component_data,
           component_labels,
           save_labels,
           mixture_data,
           marker_gene_masks=None,
           hvg_masks=None,
           max_steps_components=None,
           component_reconstruction_data=None,
           mixture_reconstruction_data=None,
           component_valid_data=None,
           component_valid_labels=None,
           mixture_valid_data=None,
           mixture_valid_labels=None,
           loadings=None):
    """Initializes and trains component VAE(s) and combined model.
    Args:
        FLAGS     (flags.FLAGS): Tensorflow flags.
        config    (ConfigProto): Hardware configuration options.
        component_data (matrix): Component VAE input data.
        component_labels  (str): Membership vector for each sample in component data.
        mixture_data   (matrix): Combined model input data to estimate proportions for.
        marker_gene_masks  (str): Genes to be used during deconvolution.
        component_reconstruction_data (matrix): Data to compare against component VAE reconstructions.
        mixture_reconstruction_data   (matrix): Data to compare against combined model reconstructions.
        component_valid_data (matrix): Component VAE validation (test) data.
        component_valid_labels  (str): Membership vector for each sample in validation (test) component data.
        mixture_valid_data (matrix): Component VAE validation (test) data.
        mixture_valid_labels  (str): Membership vector for each sample in validation (test) component data.
        loadings (matrix): Projection matrix to convert from low dimensions to gene expression space.
    Returns:
        proportions: Mixture proportions for each cell with respect to all component VAE(s).
        weights: Unnormalized mixture probabilities.
    """

    # ## Start deconvolution at top of tree (Python dictionary retains insertion order)
    # for key in component_labels.keys():
    ## Dictionary containing all VAE models named by unique elements within datasets
    VAE = {}
    ## Define network structure
    graph = tf.Graph()
    with graph.as_default():
        global_step_component = tf.train.get_or_create_global_step()
        ## Create component VAEs
        for scope in np.unique(component_labels):
            ## Set up encoder model. q(z|x)
            VAE[scope] = vaeBackend.modelVAE(FLAGS,
                                              latent_dims=FLAGS.latent_dims,
                                              input_size=component_data.shape[-1],
                                              output_size=component_data.shape[-1] if FLAGS.rec_project is False else component_reconstruction_data.shape[-1], ## Determines the output size of decoder
                                              num_samples=component_data[np.where(component_labels == scope)[0],:].shape[0],
                                              reconstruction=component_reconstruction_data,
                                              loadings=tf.cast(loadings, tf.float32) if loadings is not None else None,
                                              scope=scope,
                                              marker_masks=marker_gene_masks,
                                              hvg_masks=hvg_masks)

        ## Construct combined model for training/all mixture data which can potentially update VAEs
        if mixture_data is not None and FLAGS.deconvolution is True:

            deconvModel = deconvBackend.deconvModel(FLAGS=FLAGS,
                                                    VAE=VAE,
                                                    datasets=component_labels,
                                                    input_size=mixture_data.shape[-1],
                                                    output_size=mixture_data.shape[-1], ## Determines the output size of decoder
                                                    num_samples=mixture_data.shape[0],
                                                    reconstruction=mixture_data,
                                                    marker_gene_masks=marker_gene_masks,
                                                    loadings=tf.cast(loadings, tf.float32) if loadings is not None else None,
                                                    scope="deconvolution")

            batchModel = batchBackend.batchModel(FLAGS=FLAGS,
                                                 VAE=VAE,
                                                 mixture_weights=deconvModel.mixture_weights,
                                                 proportions=deconvModel.proportions,
                                                 datasets=component_labels,
                                                 input_size=mixture_data.shape[-1],
                                                 output_size=mixture_data.shape[-1], ## Determines the output size of decoder
                                                 num_samples=mixture_data.shape[0],
                                                 reconstruction=mixture_data,
                                                 loadings=tf.cast(loadings, tf.float32) if loadings is not None else None,
                                                 scope="batch_correction")

        ## Tensorboard summary and model checkpoint saving
        summary_op, summary_writer, summary_writer_test, saver = utils.summary_and_checkpoint(FLAGS, graph)

    ## Training scope
    with tf.Session(graph=graph, config=config) as sess:

        ## Set the logging level for tensorflow to only fatal issues
        tf.logging.set_verbosity(tf.logging.FATAL)

        ## Define seed at the graph-level
        ## From docs: If the graph-level seed is set, but the operation seed is not:
        ## The system deterministically picks an operation seed in conjunction with
        ## the graph-level seed so that it gets a unique random sequence.
        tf.set_random_seed(FLAGS.seed)

        ## Initialize everything
        tf.global_variables_initializer().run()
        print("Done random initialization")

        if not os.path.exists(os.path.abspath(FLAGS.logdir)):
           os.makedirs(os.path.abspath(FLAGS.logdir))

        ## Assert that nothing more can be added to the graph
        # tf.get_default_graph().finalize()

        ## Track epoch loss
        epoch_loss=[]

        ## Don't normalize output
        if component_reconstruction_data is None:
            component_reconstruction_data = component_data
        if mixture_reconstruction_data is None:
            mixture_reconstruction_data = mixture_data

        ## Initialize the component VAE Dataset iterators
        for scope in np.unique(component_labels):
            sess.run(VAE[scope].train_init_op, feed_dict={VAE[scope].data_ph: component_data[np.where(component_labels == scope)[0],:],
                                                          VAE[scope].rec_data_ph: component_reconstruction_data[np.where(component_labels == scope)[0],:]})

        ## Component training!
        if FLAGS.train_component is True:
            for step in range(1,FLAGS.max_steps_component+1):
                for scope in np.unique(component_labels):
                    if (step <= max_steps_components[scope]) and (VAE[scope].early_stop is not True):
                        with tf.variable_scope(scope, tf.AUTO_REUSE):
                            _, _, summaries_train, train_loss, logp, kl_loss, mse_loss = sess.run([VAE[scope].train_op, VAE[scope].update_op, VAE[scope].summary_op, VAE[scope].train_loss, VAE[scope].avg_distortion, VAE[scope].avg_KL_div, VAE[scope].mse],
                                                                                                  options=None,
                                                                                                  run_metadata=None)
                            ## Log mphate data
                            if FLAGS.m_phate_logging is True and step < 500:
                                embedding = sess.run(VAE[scope].emb_saver_sample, feed_dict={VAE[scope].fdata_ph: mixture_data})
                                resultsObj.update_mphate(scope, embedding)

                            if step % FLAGS.print_step == 0 or step == FLAGS.max_steps_component:
                                if FLAGS.component_corr_check is True:
                                    train_corr = sess.run(VAE[scope].corr)
                                    print("(%s) Step %s: %-6.2f    MSE: %-8.4f    LogP: %-8.4f    KLL %-8.4f    Corr %-8.4f" % (scope, step, train_loss, mse_loss, logp, kl_loss, train_corr))
                                else:
                                    print("(%s) Step %s: %-6.2f    MSE: %-8.4f    LogP: %-8.4f    KLL %-8.4f" % (scope, step, train_loss, mse_loss, logp, kl_loss))

                            if FLAGS.monitor_bulk_mse is True:
                                bulk_loss = VAE[scope].mixture_mse_monitor(mixture_data, FLAGS)
                                print("%-6.2f" % bulk_loss)

                            if component_valid_data is not None:
                                comp_valid_data = component_valid_data[np.where(component_valid_labels == scope)[0],:]
                                if comp_valid_data.shape[0] > 0 :
                                    ## Swap to testing data
                                    sess.run(VAE[scope].test_init_op, feed_dict={VAE[scope].data_ph: comp_valid_data,
                                                                                 VAE[scope].rec_data_ph: comp_valid_data})
                                    ## Test model
                                    if FLAGS.component_corr_check is True:
                                        summaries_test, test_loss, test_logp, test_kl_loss, test_mse_loss, test_corr = sess.run([VAE[scope].summary_op, VAE[scope].train_loss, VAE[scope].avg_distortion, VAE[scope].avg_KL_div, VAE[scope].mse, VAE[scope].corr])
                                    else:
                                        summaries_test, test_loss, test_logp, test_kl_loss, test_mse_loss, test_corr = sess.run([VAE[scope].summary_op, VAE[scope].train_loss, VAE[scope].avg_distortion, VAE[scope].avg_KL_div, VAE[scope].mse])
                                    ## Early stopping if enabled and minimum number of steps has been achieved
                                    if FLAGS.early_stopping is True and step >= FLAGS.early_min_step and FLAGS.component_corr_check is True:
                                        utils.early_stopping(FLAGS, VAE, scope, test_corr)

                                    ## Swap back to training data
                                    sess.run(VAE[scope].train_init_op, feed_dict={VAE[scope].data_ph: component_data[np.where(component_labels == scope)[0],:],
                                                                                  VAE[scope].rec_data_ph: component_reconstruction_data[np.where(component_labels == scope)[0],:]})
                            else:
                                    ## Early stopping if enabled and minimum number of steps has been achieved
                                    if FLAGS.early_stopping is True and step >= FLAGS.early_min_step and FLAGS.component_corr_check is True:
                                        train_corr = sess.run(VAE[scope].corr)
                                        utils.early_stopping(FLAGS, VAE, scope, train_corr)

                            ## Report loss
                            if step % FLAGS.log_step == 0 or step == FLAGS.max_steps_component:
                                summary_writer.add_summary(summaries_train, str(step))
                                summary_writer.flush()

                                ## Record results
                                resultsObj.update_component_metrics("train", scope, step, train_loss, mse_loss, logp, kl_loss)

                                ## Run test data if available
                                if component_valid_data is not None:
                                    if comp_valid_data.shape[0] > 0 :
                                        resultsObj.update_component_metrics("test", scope, step, test_loss, test_mse_loss, test_logp, test_kl_loss)
                                        ## Summary reports (Tensorboard), testing summary
                                        summary_writer_test.add_summary(summaries_test, str(step))
                                        summary_writer_test.flush()


                    ## Save model and summaries
                    if step % FLAGS.log_step == 0 or step == FLAGS.max_steps_component:
                        ## Summary reports (Tensorboard)
                        summary_writer.add_summary(summaries_train, str(step))
                        summary_writer.flush()
                        if component_valid_data is not None:
                            summary_writer_test.add_summary(summaries_test, str(step))
                            summary_writer_test.flush()
                        ## Write out graph
                        if step == FLAGS.max_steps_component:
                            save = saver.save(sess, os.path.abspath(FLAGS.logdir + '/ckpt/model_component.ckpt'), step)

            ## Save reconstruction results prior to mixture_stage training
            if FLAGS.log_results is True:
                saving_utils.save_results(sess=sess,
                                   resultsObj=resultsObj,
                                   FLAGS=FLAGS,
                                   VAE=VAE,
                                   component_labels=save_labels,
                                   component_data=component_data,
                                   hvg_masks=hvg_masks,
                                   marker_gene_masks=marker_gene_masks,
                                   stage="component",
                                   component_valid_labels=component_valid_labels,
                                   component_valid_data=component_valid_data,
                                   mixture_input_data=mixture_data)
            # if FLAGS.log_samples is True:
            #     saving_utils.save_samples(sess=sess,
            #                        resultsObj=resultsObj,
            #                        FLAGS=FLAGS,
            #                        VAE=VAE,
            #                        component_labels=component_labels,
            #                        stage="prebatch")
        else:
            ## Load a previously saved model
            saver.restore(sess, tf.train.latest_checkpoint(os.path.abspath(FLAGS.logdir +  '/ckpt')))

        ## Proportion training
        if mixture_data is not None and FLAGS.deconvolution is True:

            ## Initialize the mixture dataset
            sess.run(deconvModel.mix_iter_data.initializer, feed_dict={deconvModel.mix_data_ph: mixture_data})

            ## Initialize the mixture dataset
            sess.run(batchModel.mix_iter_data.initializer, feed_dict={batchModel.mix_data_ph: mixture_data})

            for step in range(1,FLAGS.proportion_steps+1):
                ## Proportion optimizer
                _, summaries, prop_loss = sess.run([deconvModel.train_proportion_op, summary_op, deconvModel.proportion_loss])

                if step % FLAGS.print_step == 0 or step == FLAGS.proportion_steps:
                    print("(Proportions) Step: %s    MSE: %-8.4f" % (step, prop_loss))

                ## Report loss
                if step % FLAGS.log_step == 0:
                    summary_writer.add_summary(summaries, str(step+FLAGS.max_steps_component))

                if step % np.ceil(mixture_data.shape[0]/FLAGS.batch_size_mixture) == 0:
                    epoch_loss = np.mean(epoch_loss)
                    resultsObj.update_mixture_metrics("train", epoch_loss, epoch_loss)
                    epoch_loss = [] ## Reset
                else:
                    epoch_loss.append(prop_loss)

                ## Save model and summaries
                if step % FLAGS.log_step == 0 or step == FLAGS.proportion_steps:
                    ## Summary reports (Tensorboard)
                    summary_writer.add_summary(summaries, str(step+FLAGS.max_steps_component))
                    ## Get estiamted proportions
                    proportions = sess.run(deconvModel.proportions)
                    #weights = sess.run(deconvModel.mixture_weights)
                    resultsObj.update_proportions(step, proportions)
                    if step == FLAGS.proportion_steps:
                        ## Write out graph
                        save = saver.save(sess, os.path.abspath(FLAGS.logdir + '/ckpt/model_combined.ckpt'), step+FLAGS.max_steps_component)

            ## Save mixture probabilities
            # proportions = sess.run(deconvModel.proportions)
            # np.savetxt(FLAGS.logdir + "/probabilities.csv",
            #            proportions,
            #            delimiter=",",
            #            header=','.join(str(x) for x in np.unique(component_labels)),
            #            comments='')

            ## Save mixture weights
            # weights = sess.run(deconvModel.mixture_weights)
            # np.savetxt(FLAGS.logdir + "/weights.csv",
            #            weights,
            #            delimiter=",",
            #            header=','.join(str(x) for x in np.unique(component_labels)),
            #            comments='')

            # deconvModel.save_mixtures(sess=sess,
            #                             resultsObj=resultsObj,
            #                             FLAGS=FLAGS,
            #                             VAE=VAE,
            #                             mixture_input_data=mixture_data,
            #                             datatype='prebatch',
            #                             batch_size=100)

            ## Mixture batch-correction training
            if FLAGS.batch_correction is True:
                ## Record the VAE weights after pretraining
                # if FLAGS.tied_model_l2 is True and step == FLAGS.max_steps_component:
                #     for scope in np.unique(component_labels):
                #         print("Assigning weights")
                #         VAE[scope].assign_network_weights(sess=sess)

                ## Train!
                for step in range(1,FLAGS.max_steps_combined+1):
                    ## Combined optimizer
                    _, summaries, train_loss, mse_loss, marker_mse, non_marker_mse = sess.run([batchModel.train_batch_op, summary_op, batchModel.train_batch_loss, batchModel.batch_loss, batchModel.marker_mse, batchModel.non_marker_mse])

                    if step % FLAGS.print_step == 0 or step == FLAGS.max_steps_combined:
                        print("(Batch Correct) Step %s: %-6.2f    LogP(mixture): %-8.4f    MSE(marker): %-8.4f    MSE(Non-marker): %-8.4f" % (step, train_loss, mse_loss, marker_mse, non_marker_mse))

                    ## Report loss
                    if step % FLAGS.log_step == 0:
                        for scope in np.unique(component_labels):
                            with tf.variable_scope(scope, tf.AUTO_REUSE):
                                train_loss, logp, kl_loss, mse_loss = sess.run([VAE[scope].train_loss, VAE[scope].avg_distortion, VAE[scope].avg_KL_div, VAE[scope].mse], options=None, run_metadata=None)
                                resultsObj.update_component_metrics("train-post", scope, step, train_loss, mse_loss, logp, kl_loss)
                                #print(sess.run(VAE[scope].data_index))
                        summary_writer.add_summary(summaries, str(step+FLAGS.max_steps_component+FLAGS.proportion_steps))
                        resultsObj.update_mixture_metrics("train", train_loss, mse_loss)

                    ## Save model and summaries
                    if step % FLAGS.log_step == 0 or step == FLAGS.max_steps_combined:
                        ## Summary reports (Tensorboard)
                        summary_writer.add_summary(summaries, str(step+FLAGS.max_steps_component+FLAGS.proportion_steps))
                        ## Get estiamted proportions
                        proportions = sess.run(batchModel.proportions)
                        #weights = sess.run(batchModel.mixture_weights)
                        resultsObj.update_proportions(step, proportions)
                        if step == FLAGS.max_steps_combined:
                            ## Write out graph
                            save = saver.save(sess, os.path.abspath(FLAGS.logdir + '/ckpt/model_combined.ckpt'), step+FLAGS.max_steps_component+FLAGS.proportion_steps)

                if FLAGS.log_results is True:
                    print("Logging combined results")
                    ## Save reconstruction results prior to mixture_stage training
                    saving_utils.save_results(sess=sess,
                                       resultsObj=resultsObj,
                                       FLAGS=FLAGS,
                                       VAE=VAE,
                                       component_labels=save_labels,
                                       component_data=component_data,
                                       hvg_masks=hvg_masks,
                                       marker_gene_masks=marker_gene_masks,
                                       stage="combined",
                                       component_valid_labels=component_valid_labels,
                                       component_valid_data=component_valid_data,
                                       mixture_input_data=mixture_data)

                # deconvModel.save_mixtures(sess=sess,
                #                             resultsObj=resultsObj,
                #                             FLAGS=FLAGS,
                #                             VAE=VAE,
                #                             mixture_input_data=mixture_data,
                #                             datatype='postbatch',
                #                             batch_size=100)

            # print(deconvModel.trainable_prop)
            # print(deconvModel.trainable_comb)
        #print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        ## Record the VAE weights after pretraining
        # if FLAGS.tied_model_l2 is True:
        #     for scope in np.unique(component_labels):
        #         print("Assigning weights")
        #         VAE[scope].assign_network_weights(sess=sess, assign=False)
    ## Return depending on deconvolution
    return resultsObj
