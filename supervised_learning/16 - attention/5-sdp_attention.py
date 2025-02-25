#!/usr/bin/env python3
"""
function that calculates the scaled dot product attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    function that calculates the scaled dot product attention
    Args:
    Q: tensor with its last two dimensions as (..., seq_len_q, dk)
        containing the query matrix
    K: tensor with its last two dimensions as (..., seq_len_v, dk)
        containing the key matrix
    V: tensor with its last two dimensions as (..., seq_len_v, dv)
        containing the value matrix
    mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
        containing the optional mask, or defaulted to None
    Returns:
    output: tensor with its last two dimensions as (..., seq_len_q, dv)
        containing the scaled dot product attention
    weights: tensor with its last two dimensions as (.., seq_len_q, seq_len_v)
        containing the attention weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)
    return output, attention_weights
