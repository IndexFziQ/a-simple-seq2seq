# xieyuqiang
# 2018-6-4

import ops
import tensorflow as tf
import data_utils
import os
import math
import time
import sys
import random
import numpy as np

from utils import function
from search import beam, select_nbest

class HyParams(object):
    def __init__(self):
        self.learning_rate_decay_factor = 0.98
        self.learning_rate = 0.0002
        self.start_decay_step = 0
        self.per_step_decay = 10000

        self.batch_size = 80
        self.dropout_rate = 0.5
        self.lp_weight = 1.0
        self.max_gradient_norm = 5.0

        self.emb_size = 620
        self.hidden_size = 1000
        self.attn_size = 1000

        self.beam_width = 10
        self.minval = -0.08
        self.maxval = 0.08

        self.source_emb = None
        self.target_emb = None


def rnn_encoder(cell, inputs, sequence_length, parallel_iterations=None,
                swap_memory=False, dtype=None):    
    parallel_iterations = parallel_iterations or 32

    batch = tf.shape(inputs)[1]
    dtype = dtype or inputs.dtype

    initial_state = cell.zero_state(batch, dtype)

    outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length,
                                        initial_state=initial_state,
                                        swap_memory=swap_memory,
                                        time_major=True, dtype=dtype,
                                        parallel_iterations=parallel_iterations)

    return outputs, state

def encoder(cell_fw, cell_bw, inputs, sequence_length,
            parallel_iterations=None, swap_memory=False,
            dtype=None, scope=None)
    time_dim = 0
    batch_dim = 1

    # variable_scope: A context manager for defining ops that creates variables (layers).
    # use variable_scope function to create shared variables implicitly
    with tf.variable_scope(scope or "encoder"):
        with tf.variable_scope("forward"):
            output_fw, state_fw = rnn_encoder(cell_fw, inputs,
                                            sequence_length,
                                            parallel_iterations,
                                            swap_memory, dtype)

        # background direction    
        inputs_reverse = tf.reverse_sequence(inputs, sequence_length, time_dim,
                                            batch_dim)

        with tf.variable_scope("background"):
            output_bw