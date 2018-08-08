# a simple seq2seq model
# author: IndexFziQ
# email: IndexFziQ@gmail.com
# It learns from pemywei's seq2seq model. The more annotations are wanted.

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
        """Initialize all hyperparameters

        Args:        
            learning_rate: learning rate to start with
            learning_rate_decay_factor: decay learning rate by this much when needed
            
            To change the learning rate with the step 10000

            batch_size: the size of the batches used during training
            dropout_rate: the rate of dropout
            lp_weight: the weight of layout
            max_gradient_norm: gradients will be clipped to maximally this norm

            emb_size: the dimension of embedding
            hidden_size: the number of units in each layer of the model
            attn_size: the dimension of attention vector

            beam_width: the width of beam search
            minval: a python scalar or a scalar tensor. Lower bound of the range of random values to generate.
            maxval: a python scalar or a scalar tensor. Upper bound of the range of random values to generate.  Defaults to 1 for float types.

            source_emb: the size of the source vocabulary
            target_emb: the size of the target vocabulary                                            
        """
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

    # use zero_state function to get the initial state
    initial_state = cell.zero_state(batch, dtype)
    """Creates a recurrent neural network specified by RNNCell `cell`.

    Function: 
        tf.nn.dynamic_rnn   
    Args:
        cell: An instance of RNNCell.
        inputs: The RNN inputs.    
        sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
        initial_state: (optional) An initial state for the RNN.      
        dtype: (optional) The data type for the initial state and expected output.
        parallel_iterations: (Default: 32).  The number of iterations to run in parallel.
        swap_memory: Transparently swap the tensors produced in forward inference 
                    but needed for back prop from GPU to CPU.

    Return:
        A pair (outputs, state)
        outputs: The RNN output `Tensor`.
        state: The final state.
    """
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length,
                                        initial_state=initial_state,
                                        swap_memory=swap_memory,
                                        time_major=True, dtype=dtype,
                                        parallel_iterations=parallel_iterations)

    return outputs, state

def encoder(cell_fw, cell_bw, inputs, sequence_length,
            parallel_iterations=None, swap_memory=False, dtype=None, 
            scope=None):
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
        # tf.reverse_sequence: everses variable length slices.    
        inputs_reverse = tf.reverse_sequence(inputs, sequence_length, time_dim,
                                            batch_dim)

        with tf.variable_scope("background"):
            output_bw, state_bw = rnn_encoder(cell_bw, inputs_reverse, 
                                            sequence_length, 
                                            parallel_iterations, 
                                            swap_memory, dtype)
            
            output_bw = tf.reverse_sequence(output_bw, sequence_length, time_dim,
                                            time_dim, batch_dim)

    return tf.concat([output_fw, output_bw], 2)

def attention(query, attention_states, mapped_states, attention_mask, 
              attn_size, dtype=None, scope=None):
    with tf.variable_scope(scope or "attention", dtype=dtype):
        """
            get_shape(): Get alias of TensorShape.
            as_list(): Get the size of TensorShape, shape=[d1 d2]
            tf.shape(): Returns the TensorShape that represents the shape of this tensor.
        """
        hidden_size = attention_states.get_shape().as_list()[2]
        shape = tf.shape(attention_states)
        
        """
            tf.reshape(): Given tensor, this operation returns a tensor that has the same values as tensor with shape given.
            https://tensorflow.google.cn/versions/master/api_docs/python/tf/reshape
        """
        if mapped_states is None:
            batched_states = tf.reshape(attention_states,[-1, hidden_size])
            mapped_states = ops.nn.linear(batched_states, attn_size, True,
                                        scope="states")
            mapped_states = tf.reshape(mapped_states,
                                        [shape[0],shape[1],attn_size])
            
            if query is None:
                return mapped_states
        
        mapped_query = ops.nn.linear(query, attn_size, False, scope="logits")
        mapped_query = mapped_query[None, :, :]

        # Computes hyperbolic tangent of `x` element-wise.
        hidden = tf.tanh(mapped_query + mapped_states)
        hidden = tf.reshape(hidden, [-1, attn_size])

        score = ops.nn.linear(hidden, 1, True, scope="hidden")
        # Computes exponential of x element-wise. \(y = e^x\).
        exp_score = tf.exp(score)
        exp_score = tf.reshape(exp_score, [shape[0], shape[1]])

        if attention_mask is not None:
            exp_score = exp_score * attention_mask
        
        # Computes the sum of elements across dimensions of a tensor. 
        # https://tensorflow.google.cn/api_docs/python/tf/reduce_sum
        alpha = exp_score / tf.reduce_sum(exp_score, 0)[None, :]
    
    return alpha[:, :, None]

def decoder(cell, inputs, initial_state, attention_states, attention_length,
            attention_size, dtype=None, scope=None):
    if inputs is None:
        raise ValueError("inputs must not be None!")

    time_steps = tf.shape(inputs)[0]
    batch = tf.shape(inputs)[1]
    output_size = cell.output_size
    dtype = dtype or inputs.dtype
    # Returns a mask tensor representing the first N positions of each cell.
    # https://tensorflow.google.cn/api_docs/python/tf/sequence_mask
    attention_mask = tf.sequence_mask(attention_length, dtype=dtype)
    attention_mask = tf.transpose(attention_mask)

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    with tf.variable_scope(scope or "decoder", dtype=dtype):
        mapped_states = attention(None, attention_states, None, None,
                                attention_size)
        
        """Construct a new TensorArray or wrap an existing TensorArray handle.

        Args:
            tensor_array_name: (optional) Python string: the name of the TensorArray.
                This is used when creating the TensorArray handle. 
            time_steps: 
            
        """
        input_ta = tf.TensorArray(tf.float32, time_steps,
                                    tensor_array_name="input_array")
        output_ta = tf.TensorArray(tf.float32, time_steps,
                                    tensor_array_name="output_array")
        context_ta = tf.TensorArray(tf.float32, time_steps,
                                    tensor_array_name="context_array")
        # Unpacks the given dimension of a rank=R tensor into rank=(R-1) tensors.
        input_ta = input_ta.unstack(inputs)

        def loop(time, output_ta, context_ta, state):
            inputs = input_ta.read(time)

            with tf.variable_scope("below"):
                output, state = cell(inputs, state)

            alpha = attention(output, attention_size, mapped_states,
                            attention_mask, attention_size)
            context = tf.reduce_sum(alpha * attention_states, 0)

            with tf.variable_scope("above"):
                output, new_state = cell(context, state)
            output_ta = output_ta.write(time, output)
            context_ta = context_ta.write(time, context)
            return time + 1, output_ta, context_ta, new_state

        # Creates a constant tensor.
        time = tf.constant(0, dtype=tf.int32,name="time")
        cond = lambda time, *_: time < time_steps
        loop_vars = (time, output_ta, context_ta, initial_state)

        # Repeat body while the condition cond is true.
        outputs = tf.while_loop(cond, loop, loop_vars, parallel_iterations=32,
                                swap_memory=True)
        
        time, output_final_ta, context_final_ta, final_state = outputs

        final_output = output_final_ta.stack()
        final_context = context_final_ta.stack()
        final_output.set_shape([None, None, output_size])
        final_context.set_shape([None, None, 2 * output_size])

    return final_output, final_context

class NMT(object):

    def __init__(self, hyparams, svocab_size, tvocab_size, eos="</s>", unk="UNK", is_training=True):
        self.learning_rate = hyparams.learning_rate
        self.global_step = tf.Variable(0, trainable=False)
        keep_prob = 1.0 - hyparams.dropout_rate

    def prediction(prev_inputs, states, context, keep_prob=1.0):
        if states.get_shape().ndims == 3:
            states = tf.reshape(states, [-1, hyparams.hidden_size])

        if prev_inputs.get_shape().ndims == 3:
            prev_inputs = tf.reshape(prev_inputs, [-1, hyparams.emb_size])

        if context.get_shape().ndims == 3:
            context = tf.reshape(context, [-1, 2 * hyparams.hidden_size])
        
        features = [states, prev_inputs, context]
        readout = ops.nn.linear(features, hyparams.emb_size, True,
                                multibias=True, scope="deepout")
        readout = tf.tanh(readout)

        if keep_prob < 1.0:
            readout = tf.nn.dropout(readout, keep_prob=keep_prob)
        
        logits = ops.nn.linear(readout, tvocab_size, True, scope="logits")
    
        return logits

    initializer = tf.random_uniform_initializer(hyparams.minval, hyparams.maxval)

    # training graph
    with tf.variable_scope("rnnsearch", initializer=initializer):
        self.src_seq = tf.placeholder(tf.int32, [None, None], "source_sequence")
        self.src_len = tf.placeholder(tf.int32, [None], "source_length")
        self.tgt_seq = tf.placeholder(tf.int32, [None, None], "target_sequence")
        self.tgt_len = tf.placeholder(tf.int32, [None], "target_length")

    with tf.device("/cpu:0"):
        source_embedding = tf.get_variable("source_embedding",
                                            [svocab_size, hyparams.emb_size],
                                            tf.float32)
        target_embedding = tf.get_variable("target_embedding",
                                            [tvocab_size,hyparam.emb_size],
                                            tf.floats32)
        source_inputs = tf.gather(source_embedding, self.src_seq)
        target_inputs = tf.gather(target_embedding, self.tgt_seq)

        if keep_prob < 1.0:
            source_inputs = tf.nn.dropout(source_inputs, keep_prob)
            target_inputs = tf.nn.dropout(target_inputs, keep_prob)

    cell = ops.rnn_cell.GRUCell(hyparams.hidden_size)
    annotation = encoder(cell, cell, source_inputs, self.src_len)

    with tf.variable_scope("decoder"):
        ctx_sum = tf.reduce_sum(annotation, 0)
        initial_state = ops.nn.linear(ctx_sum, hyparams.hidden_size,True, 
                                    scope="initial")
        initial_state = tf.tanh(initial_state)

    zero_embbedding = tf.zeros([1, tf.shape(self.tgt_seq)[1], hyparams.emb_size])
    shift_inputs = tf.concat([zero_embbedding, target_inputs], 0)
    shift_inputs = shift_inputs[:-1, :, :]
    shift_inputs.set_shape([None, None, hyparams.emb_size])

    cell = ops.rnn_cell.GRUCell(hyparams.hidden_size)

    decoder_outputs = decoder(cell, shift_inputs, initial_state,
                                annotation, self.src_len, hyparams.attn_size)
    output, context = decoder_outputs

    with tf.variable_scope("decoder"):
        logits = prediction(shift_inputs, output, context,
                            keep_prob=keep_prob)
    
    labels = tf.reshape(self.tgt_seq, [-1])
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=labels)
    ce = tf.reshape(ce, tf.shape(self.tgt_seq))
    mask = tf.sequence_mask(self.tgt_len, dtype=tf.float32)
    mask = tf.transpose(mask)
    cost = tf.reduce_mean(tf.reduce_sum(ce * mask), 0)
    
    self.cross_entropy_loss = cost
    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()

    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    gradients = tf.gradients(self.cross_entropy_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, hyparams.max_gradient_norm)
    self.updates = self.optimizer.apply_gradients(zip(clipped_gradients, params), 
                                                    global_step=self.global_step)
    self.saver = tf.train.Saver(tf.global_variables())

    #decoding graph
    with tf.variable_scope("rnn_search",reuse=True) as scope:
        prev_word = tf.placeholder(tf.int32, [None], "prev_token")

    with tf.device("/cpu0:"):
        source_embedding = tf.get_variable("source_embedding",
                                            [svocab_size, hyparams.emb_size],
                                            tf.float32)
        target_embedding = tf.get_variable("target_embedding",
                                            [tvocab_size, hyparams.emb_size],
                                            tf.float32)
        source_inputs = tf.gather(source_embedding, self.src_seq)
        target_inputs = tf.gather(target_embedding, prev_word)
    cond = tf.equal(prev_word, 0)
    cond = tf.cast(cond, tf.float32)
    target_inputs = target_inputs * (1.0 - tf.expand_dims(cond, 1))

    #encoder
    cell = ops.rnn_cell.GRUCell(hyparams.hidden_size)
    annotation = encoder(cell, cell, source_inputs, self.src_len)

    #decoder
    with tf.variable_scope("decoder"):
        ctx_sum = tf.reduce_sum(annotation, 0)
        initial_state = ops.nn.linear(ctx_sum, hyparams.hidden_size, True,
                                        scope="initial")
        initial_state = tf.tanh(initial_state)
    
    with tf.variable_scope("decoder"):
        mask = tf.sequence_mask(self.src, tf.shape(self.src_seq)[0],
                                dtype=tf.float32)
        mask = tf.transpose(mask)
        mapped_states = attention(None, annotation, None, None,
                                hyparams.attn_size)
    
    cell = ops.rnn_cell.GRUCell(hyparams.hidden_size)

    with tf.variable_scope("decoder"):
        with tf.variable_scope("below"):
            output, state = cell(target_inputs, initial_state)
        alpha = attention(output, annotation, mapped_states,
                          mask, hyparams.attn_size)
        context = tf.reduce_sum(alpha * annotation, 0)
        
        with tf.variable_scope("above"):
            output, next_state = cell(context, state)
        logits = prediction(target_inputs, next_state, context)
        probs = tf.nn.softmax(logits)

    encoding_inputs = [self.src_seq, self.src_len]
    encoding_outputs = [annotation, mapped_states, initial_state, mask]
    encode = function(encoding_inputs, encoding_outputs)

    prediction_inputs = [prev_word, initial_state, annotation,
                        mapped_states, mask]
    prediction_outputs = [probs, next_state, alpha]
    predict = function(prediction_inputs, prediction_outputs)

    self.encode = encode
    self.predict = predict

    def train_step(self, sess, train_set, svocab, tvocab, unk, iteration):
        sbatch_data, sdata_length, tbatch_data, tdata_length = \
                data_utils.nextBatch(train_set, svocab, tvocab, unk, 
                                    iteration * self.batch_size, self.batch_size)
        _, step_loss = \
            sess.run(
                [
                    self.updates,
                    self.cross_entropy_loss,
                ],
                feed_dict = {
                    self.src_seq: sbatch_data,
                    self.src_len: sdata_length,
                    self.tgt_seq: tbatch_data,
                    self.tgt_len: tdata_length
                }
            )
        return step_loss * len(tdata_length) / sum(tdata_length)

def beamsearch(model, tvocab, itvocab, eos_symbol, seq, seqlen=None,
               beamsize=10, normalize=False, maxlen=None, minlen=None):
    size = beamsize
    encode = model.encode
    predict = model.predict

    eosid = tvocab[eos_symbol]

    time_dim = 0
    batch_dim = 1

    if seqlen is None:
        seq_len = np.array([seq.shape[time_dim]])
    else:
        seq_len = seqlen
    
    if maxlen is None:
        maxlen = seq_len[0] * 3

    if minlen is None:
        minlen = seq_len[0] / 2
    
    annotation, mapped_states, initial_state, attn_mask = encode(seq, seq_len)
    state = initial_state

    initial_beam = beam(size)
    initial_beam.candidate = [[eosid]]
    initial_beam.score = np.zeros([1], "float32")

    hypo_list = []
    beam_list = [initial_beam]

    for k in range(maxlen):
        if size == 0:
            break

        prev_beam = beam_list[-1]
        candidate = prev_beam.candidate
        num = len(prev_beam.candidate)
        last_words = np.array(map(lambda t: t[-1], candidate), "int32")

        batch_annot = np.repeat(annotation, num, batch_dim)
        batch_mannot = np.repeat(mapped_states, num, batch_dim)
        batch_mask = np.repeat(attn_mask, num, batch_dim)

        prob_dist, state, alpha = predict(last_words, state, batch_annot,
                                          batch_mannot, batch_mask)

        logprobs = np.log(prob_dist)

        if k < minlen:
            logprobs[:, eosid] = -np.inf

        # force to add eos symbol
        if k == maxlen - 1:
            # copy
            eosprob = logprobs[:, eosid].copy()
            logprobs[:, :] = -np.inf
            logprobs[:, eosid] = eosprob

        next_beam = beam(size)
        outputs = next_beam.prune(logprobs, lambda x: x[-1] == eosid,
                                  prev_beam)

        hypo_list.extend(outputs[0])
        batch_indices, word_indices = outputs[1:]
        size -= len(outputs[0])
        state = select_nbest(state, batch_indices)
        beam_list.append(next_beam)

    # postprocessing
    if len(hypo_list) == 0:
        score_list = [0.0]
        hypo_list = [[eosid]]
    else:
        score_list = [item[1] for item in hypo_list]
        # exclude bos symbol
        hypo_list = [item[0][1:] for item in hypo_list]

    for i, (trans, score) in enumerate(zip(hypo_list, score_list)):
        count = len(trans)
        if count > 0:
            if normalize:
                score_list[i] = score / count
            else:
                score_list[i] = score

    # sort
    hypo_list = np.array(hypo_list)[np.argsort(score_list)]
    score_list = np.array(sorted(score_list))

    output = []

    for trans, score in zip(hypo_list, score_list):
        trans = map(lambda x: itvocab[x], trans)
        output.append((trans, score))

    return output