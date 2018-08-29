import tensorflow as tf

from utils import get_module_functions
from utils_tf import pad_left


def _bgan_discr_loss(D_real, D_fake):
    D_loss = tf.reduce_mean(tf.nn.softplus(-D_real)) + \
           tf.reduce_mean(tf.nn.softplus(-D_fake)) + tf.reduce_mean(D_fake)
    with tf.name_scope("D_loss"):
        loss_summary = tf.summary.scalar("batch-by-batch", D_loss)
        p_real = tf.reduce_mean(tf.sigmoid(D_real))
        p_real_summary = tf.summary.scalar("p-real-1", p_real)

    return D_loss, tf.summary.merge([loss_summary, p_real_summary])


def _bgan_gen_loss(G_sample, D_fake, g_output_logit, n_samples, bs):
    # Porting of thirdparties/rdevon/BGAN/lib/loss.py
    # Details can be found in references/gen_loss_derivation.jpg

    log_w = tf.reshape(D_fake, [n_samples, bs])  # (20, 64)
    # notes.pdf, Appendix B, penultimate calculation
    log_g = -tf.reduce_sum((1. - G_sample) * pad_left(g_output_logit) +
                           pad_left(tf.nn.softplus(-g_output_logit)),
                           axis=[2, 3, 4])  # (20, 32)
    log_N = tf.log(tf.cast(log_w.shape[0], dtype=tf.float32))  # ()
    log_Z_est = tf.reduce_logsumexp(log_w - log_N, axis=0)  # (64,)
    log_w_tilde = log_w - pad_left(log_Z_est) - log_N  # (20, 64)
    w_tilde = tf.exp(log_w_tilde)  # (20, 64)

    G_loss = -tf.reduce_mean(tf.reduce_sum(w_tilde * log_g, 0))
    with tf.name_scope("G_loss"):
        loss_summary = tf.summary.scalar("batch-by-batch", G_loss)
        p_fake = 1 - tf.reduce_mean(tf.sigmoid(D_fake))
        p_fake_summary = tf.summary.scalar("p-fake-0", p_fake)

    return G_loss, tf.summary.merge([loss_summary, p_fake_summary])


def get_losses(experiment, G_sample, D_real, D_fake, g_output_logit, n_samples,
               bs):
    losses = get_module_functions(__name__)

    discriminator_loss = losses[experiment.discriminator_loss]
    D_loss, D_summaries = discriminator_loss(D_real, D_fake)

    generator_loss = losses[experiment.generator_loss]
    G_loss, G_summaries = generator_loss(
        G_sample, D_fake, g_output_logit, n_samples, bs)

    return D_loss, D_summaries, G_loss, G_summaries
