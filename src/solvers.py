import tensorflow as tf


def get_solvers(experiment, D_loss, theta_D, G_loss, theta_G):
    def adam():
        return tf.train.AdamOptimizer(learning_rate=experiment.learning_rate,
                                      beta1=0.5)
    # the moving_mean and moving_variance of batch normalization layers need to
    # be updated during training. Ref:
    # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization

    # discriminator solver (no batch normalization)
    D_solver = adam().minimize(D_loss, var_list=theta_D)

    # generator solver (batch normalization is mandatory)
    experiment.logger.debug(
        "Adding batch normalization moving_mean and moving_variance to the "
        "optimizer for generator")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        G_solver = adam().minimize(G_loss, var_list=theta_G)

    return D_solver, G_solver
