import tensorflow as tf

def nt_xent_loss(z_i, z_j, temperature=0.5):
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)
    batch_size = tf.shape(z_i)[0]
    representations = tf.concat([z_i, z_j], axis=0)
    similarity_matrix = tf.matmul(representations, representations, transpose_b=True)
    labels = tf.range(batch_size)
    labels = tf.concat([labels, labels], axis=0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, similarity_matrix / temperature, from_logits=True)
    return tf.reduce_mean(loss)

