from typing import Optional

import math
import random
import torch
from joeynmt.decoders import RecurrentDecoder
from torch import Tensor


class CustomRecurrentDecoder(RecurrentDecoder):
    def __init__(self, *args, **kwargs):
        super(CustomRecurrentDecoder, self).__init__(*args, **kwargs)

        encoder = kwargs.get('encoder')
        hidden_size = kwargs.get('hidden_size')

        self.bridge_layer_h = torch.nn.Linear(encoder.output_size, hidden_size, bias=True)
        self.bridge_layer_c = torch.nn.Linear(encoder.output_size, hidden_size, bias=True)

    def _init_hidden(self, encoder_final: Tensor = None) -> (Tensor, Optional[Tensor]):
        hidden_h = torch.tanh(self.bridge_layer_h(encoder_final)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        hidden_c = torch.tanh(self.bridge_layer_c(encoder_final)).unsqueeze(0).repeat(self.num_layers, 1, 1)

        return hidden_h, hidden_c

    def forward(self,
                trg_embed: Tensor,
                encoder_output: Tensor,
                encoder_hidden: Tensor,
                src_mask: Tensor,
                unroll_steps: int,
                hidden: Tensor = None,
                prev_att_vector: Tensor = None,
                **kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
         Unroll the decoder one step at a time for `unroll_steps` steps.
         For every step, the `_forward_step` function is called internally.
         During training, the target inputs (`trg_embed') are already known for
         the full sequence, so the full unrol is done.
         In this case, `hidden` and `prev_att_vector` are None.
         For inference, this function is called with one step at a time since
         embedded targets are the predictions from the previous time step.
         In this case, `hidden` and `prev_att_vector` are fed from the output
         of the previous call of this function (from the 2nd step on).
         `src_mask` is needed to mask out the areas of the encoder states that
         should not receive any attention,
         which is everything after the first <eos>.
         The `encoder_output` are the hidden states from the encoder and are
         used as context for the attention.
         The `encoder_hidden` is the last encoder hidden state that is used to
         initialize the first hidden decoder state
         (when `self.init_hidden_option` is "bridge" or "last").
        :param trg_embed: emdedded target inputs,
            shape (batch_size, trg_length, embed_size)
        :param encoder_output: hidden states from the encoder,
            shape (batch_size, src_length, encoder.output_size)
        :param encoder_hidden: last state from the encoder,
            shape (batch_size x encoder.output_size)
        :param src_mask: mask for src states: 0s for padded areas,
            1s for the rest, shape (batch_size, 1, src_length)
        :param unroll_steps: number of steps to unrol the decoder RNN
        :param hidden: previous decoder hidden state,
            if not given it's initialized as in `self.init_hidden`,
            shape (num_layers, batch_size, hidden_size)
        :param prev_att_vector: previous attentional vector,
            if not given it's initialized with zeros,
            shape (batch_size, 1, hidden_size)
        :return:
            - outputs: shape (batch_size, unroll_steps, vocab_size),
            - hidden: last hidden state (num_layers, batch_size, hidden_size),
            - att_probs: attention probabilities
                with shape (batch_size, unroll_steps, src_length),
            - att_vectors: attentional vectors
                with shape (batch_size, unroll_steps, hidden_size)
        """

        # do we use scheduled sampling?
        scheduled_sampling = kwargs.get('scheduled_sampling', False)
        k = kwargs.get('k', 1)
        batch_no = kwargs.get('batch_no', 0)
        embeddings = kwargs.get('embeddings', None)

        # shape checks
        self._check_shapes_input_forward(
            trg_embed=trg_embed,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            hidden=hidden,
            prev_att_vector=prev_att_vector)

        # initialize decoder hidden state from final encoder hidden state
        if hidden is None:
            hidden = self._init_hidden(encoder_hidden)

        # pre-compute projected encoder outputs
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        if hasattr(self.attention, "compute_proj_keys"):
            self.attention.compute_proj_keys(keys=encoder_output)

        # here we store all intermediate attention vectors (used for prediction)
        att_vectors = []
        att_probs = []

        batch_size = encoder_output.size(0)

        if prev_att_vector is None:
            with torch.no_grad():
                prev_att_vector = encoder_output.new_zeros(
                    [batch_size, 1, self.hidden_size])

        # unroll the decoder RNN for `unroll_steps` steps
        for i in range(unroll_steps):
            epsilon = 1
            if scheduled_sampling and i > 0:
                epsilon = k / (k + math.exp(batch_no / k))
            if random.uniform(0, 1) <= epsilon:
                prev_embed = trg_embed[:, i].unsqueeze(1)  # batch, 1, emb
            else:
                prev_output = torch.argmax(self.output_layer(prev_att_vector), dim=-1)
                prev_embed = embeddings(prev_output.long())

            prev_att_vector, hidden, att_prob = self._forward_step(
                prev_embed=prev_embed,
                prev_att_vector=prev_att_vector,
                encoder_output=encoder_output,
                src_mask=src_mask,
                hidden=hidden)
            att_vectors.append(prev_att_vector)
            att_probs.append(att_prob)

        att_vectors = torch.cat(att_vectors, dim=1)
        # att_vectors: batch, unroll_steps, hidden_size
        att_probs = torch.cat(att_probs, dim=1)
        # att_probs: batch, unroll_steps, src_length
        outputs = self.output_layer(att_vectors)
        # outputs: batch, unroll_steps, vocab_size
        return outputs, hidden, att_probs, att_vectors
