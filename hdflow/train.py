#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging
import sys

def train(variables, loss_function, dataset, optimizer, ckpt_manager,
          test_dataset=None, n_epochs=100, clip=None, logfile='log_train', ckpt_skip=10,
          **kwargs):
    """
    Custom training loop to fetch batches from datasets with different sizes
    and perform updates on variables.

    Parameters
    ----------
    variables: list
        List of trainable variables or list of lists of trainable variables.
    loss_function:
        Function that returns list of losses.
    dataset: tf.data.Dataset
        Training dataset.
    optimizer: tf.keras.optimizers.Optimizer, list
        Optimizer or list of Optimizers.
    ckpt_manager: tf.train.CheckpointManager
        Checkpoint manager for saving weights.
    test_dataset: tf.data.Dataset, optional.
        Testing dataset.
    n_epochs: int, optional
        Number of epochs. Default: 100.
    clip: int or float, optional
        Global norm value to clip gradients. Default: None.
    logfile: str, optional
        Output file for logging training statistics. Default: 'log_train'.
    ckpt_skip: int, optional
        Save weights every `ckpt_skip` epochs. Default: 10.
    **kwargs:
        kwargs to pass to loss_function.

    Returns
    -------
    None
    """
    # Reset logging file
    logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO, force=True)

    # Evaluate loss function on batch in order to get number of losses
    batch = dataset.train_batch() | kwargs
    losses = loss_function(**batch)
    n_loss = len(losses)
    dataset.reset_training()

    # Check if list of optimizers have been provided
    multi_opt = False
    if type(optimizer) == list:
        multi_opt = True
        n_opt = len(optimizer)
        update_fns = []
        for v, o in zip(variables, optimizer):
            update_fns.append(tf.function(update_function))
    else:
        update = tf.function(update_function)

    # Loop over epochs
    train_epoch = np.zeros((n_epochs, n_loss))
    test_epoch = np.zeros((n_epochs, n_loss))
    for epoch in tqdm(range(n_epochs), desc='Epoch loop'):

        # Loop over batches and update variables, keeping batch of training stats
        train = np.zeros((dataset.n_batches, n_loss))
        if multi_opt:
            for cnt in range(dataset.n_batches):
                batch = dataset.train_batch() | kwargs
                for v, o, update_fn in zip(variables, optimizer, update_fns):
                    losses = update_fn(v, loss_function, o, batch)
                train[cnt, :] = [value.numpy() for value in losses]
        else:
            for cnt in range(dataset.n_batches):
                batch = dataset.train_batch() | kwargs
                losses = update(variables, loss_function, optimizer, batch)
                train[cnt, :] = [value.numpy() for value in losses]

        # Compute mean train loss
        train = np.mean(train, axis=0).tolist()

        # Evalute losses on test batch
        try:
            if test_dataset is not None:
                batch = test_dataset.train_batch()
            else:
                batch = dataset.test_batch()
            losses = loss_function(**(batch | kwargs))
            test = [value.numpy() for value in losses]
        except ValueError:
            test = train

        # Reshuffle
        dataset.reset_training()

        # Store in epoch arrays
        train_epoch[epoch, :] = train
        test_epoch[epoch, :] = test

        # Write stats to logfile
        out = '%d ' + '%15.10f ' * 2 * n_loss
        logging.info(out % tuple([epoch] + train + test))
        sys.stdout.flush()

        # Periodically save checkpoint
        if epoch > 0 and epoch % ckpt_skip == 0:
            ckpt_manager.save()

    # Save final checkpoint
    ckpt_manager.save()

    # Return stats
    return train_epoch, test_epoch

def update_function(variables, loss_function, optimizer, batch, clip=None):
    """

    Parameters
    ----------
    variables: list
        List of trainable variables.
    loss_function:
        Function that returns list of losses.
    optimizer: tf.keras.optimizers.Optimizer
        Optimizer.
    fn_args:
        Arguments passed to loss function.
    batch:
        tf.data.Dataset batch passed to loss function.
    clip: int, float, or None
        Global norm value to clip gradients. Default: None.

    Returns
    -------
    losses: list
        List of scalar losses.
    """
    # Compute gradient of total loss
    with tf.GradientTape() as tape:
        losses = loss_function(**batch)
        total_loss = sum(losses)
    grads = tape.gradient(total_loss, variables)

    # Clip gradients
    if clip is not None:
        grads, _ = tf.clip_by_global_norm(grads, clip)

    # Apply gradients
    optimizer.apply_gradients(zip(grads, variables))

    # Return all losses
    return losses

def create_checkpoint_manager(checkdir, restore=False, max_to_keep=1, **kwargs):
    """
    Convenience function for creating a checkpoint manager for loading and
    saving model weights during training.

    Parameters
    ----------
    checkdir: str
        Checkpoint directory to save to and load from.
    restore: bool
        Restore previously saved weights. Default: False.
    max_to_keep: int
        Maximum number of checkpoints to save in checkdir. Default: 1.
    **kwargs:
        kwargs passed to tf.train.Checkpoint.

    Returns
    -------
    ckpt_manager: tf.train.CheckpointManager
        The checkpoint manager.
    """
    ckpt = tf.train.Checkpoint(**kwargs)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkdir, max_to_keep)
    if restore:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    return ckpt_manager


# end of file
