# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from typing import Optional, Dict, List, Any, Tuple

from omegaconf import DictConfig
from argparse import Namespace

import torch
from torch import nn, Tensor
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
    transformer_wmt_en_de_big,
    transformer_iwslt_de_en,
    transformer_mbart_large, TransformerDecoder)

logger = logging.getLogger(__name__)

@register_model("prefix_transformer")
class PrefixTransformer(TransformerModel):
    """
    See "The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)"
    """

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(PrefixTransformer, PrefixTransformer).add_args(parser)
        # Prompt tuning
        parser.add_argument('--encoder-prefix-init', type=str, metavar='N', default='from-vocab',
                            help='encoder prefix embedding init method [from-vocab, uniform]')
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.args = args
        for n, p in self.named_parameters():
            if 'prefix_' not in n:
                p.requires_grad = False

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        prefix_transformer(args)

        return super().build_model(args, task)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return PrefixTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False))
        
    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        prefix_length = args.encoder_prefix_length + args.decoder_prefix_length
        num_embeddings = len(dictionary) - prefix_length
        padding_idx = dictionary.pad()

        emb = EmbeddingsWithPrefixes(num_embeddings, embed_dim, padding_idx, args.encoder_prefix_length, args.decoder_prefix_length)
        return emb

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None):

        # Do not enforce the key match due to the prefix params
        strict = False

        # initialization of prefix tokens
        state = super().load_state_dict(state_dict, strict)

        logger.info(f'missing keys: {state.missing_keys}')
        logger.info(f'missing keys: {state.unexpected_keys}')

        token_embeddings = state_dict['encoder.embed_tokens.weight']
        self.encoder.embed_tokens.token_embeddings.load_state_dict({'weight':token_embeddings}, True)

        if self.args.encoder_prefix_init == 'from-vocab':
            sample_tokens_idx = random.sample(range(0, token_embeddings.shape[0]), self.encoder.embed_tokens.prefix_embeddings.weight.shape[0])
            prefix_dict = {'weight': token_embeddings[sample_tokens_idx]}
            self.encoder.embed_tokens.prefix_embeddings.load_state_dict(prefix_dict, True)
            logger.info(f'encoder.embed_tokens.prefix_embeddings.weight is initialized from vocabulary')
        else:
            logger.info(f'encoder.embed_tokens.prefix_embeddings.weight is uniformly initialized')

        # Zero-ing the padding idx
        nn.init.constant_(
            self.encoder.embed_tokens.token_embeddings.weight[self.encoder.embed_tokens.token_embeddings.padding_idx],
            0)
        nn.init.constant_(
            self.encoder.embed_tokens.prefix_embeddings.weight[self.encoder.embed_tokens.prefix_embeddings.padding_idx],
            0)

        return state


class PrefixLayers(nn.Module):

    def __init__(self, prefix_length, hidden_dim):
        super(PrefixLayers, self).__init__()
        self.prompt_length = prefix_length
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.zeros(prefix_length, hidden_dim))
        self.init_weights()

    def init_weights(self, range=None):
        self.weight.data.normal_(mean=0, std=self.hidden_dim ** -0.5)

    def get_prefix_length(self):
        return self.prefix_length

    def forward(self, bsz):
        return self.weight.unsqueeze(0).repeat(bsz,1,1)


class EmbeddingsWithPrefixes(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx, encoder_prefix_length, decoder_prefix_length):
        super(EmbeddingsWithPrefixes, self).__init__()
        self.encoder_prefix_length = encoder_prefix_length
        self.decoder_prefix_length = decoder_prefix_length
        self.prefix_length = encoder_prefix_length + encoder_prefix_length
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.prefix_padding_idx = 0
        self.token_embeddings = nn.Embedding(self.num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.prefix_embeddings = nn.Embedding(self.prefix_length+1, embedding_dim, padding_idx=self.prefix_padding_idx)
        self.init_weights()

    def init_weights(self, range=None):
        nn.init.normal_(self.token_embeddings.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.token_embeddings.weight[self.padding_idx], 0)
        nn.init.normal_(self.prefix_embeddings.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.prefix_embeddings.weight[self.prefix_padding_idx], 0)

    def get_prefix_length(self):
        return self.prefix_length

    def forward(self, input):
        token_input = input.detach().clone()
        prefix_input = input.detach().clone()
        prefix_mask = input > self.num_embeddings - 1
        prefix_input[~prefix_mask] = self.prefix_padding_idx
        prefix_input[prefix_mask] = prefix_input[prefix_mask] - self.num_embeddings + 1
        token_input[prefix_mask] = self.padding_idx
        prefix_embs = self.prefix_embeddings(prefix_input)
        token_embs = self.token_embeddings(token_input)

        return prefix_embs + token_embs


class PrefixTransformerDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)

    def get_output_projection_weight(self):
        return self.embed_tokens.token_embeddings.weight


@register_model_architecture("prefix_transformer", "prefix_transformer")
def prefix_transformer(args):
    args.encoder_prefix_length = getattr(args, "encoder_prefix_length", 20)
    args.decoder_prefix_length = getattr(args, "decoder_prefix_length", 20)
    args.ignore_prefix_size = args.decoder_prefix_length
    args.encoder_prefix_init = getattr(args, "encoder_prefix_init", "from-vocab")
    base_architecture(args)


@register_model_architecture("prefix_transformer", "prefix_transformer_wmt_en_de_big")
def prefix_transformer_wmt_en_de_big(args):
    args.encoder_prefix_length = getattr(args, "encoder_prefix_length", 20)
    args.decoder_prefix_length = getattr(args, "decoder_prefix_length", 20)
    args.ignore_prefix_size = args.decoder_prefix_length
    args.encoder_prefix_init = getattr(args, "encoder_prefix_init", "from-vocab")
    transformer_wmt_en_de_big(args)


@register_model_architecture("prefix_transformer", "prefix_transformer_iwslt_de_en")
def prefix_transformer_iwslt_de_en(args):
    args.encoder_prefix_length = getattr(args, "encoder_prefix_length", 20)
    args.decoder_prefix_length = getattr(args, "decoder_prefix_length", 20)
    args.ignore_prefix_size = args.decoder_prefix_length
    args.encoder_prefix_init = getattr(args, "encoder_prefix_init", "from-vocab")
    transformer_iwslt_de_en(args)


@register_model_architecture("prefix_transformer", "prefix_transformer_mbart_large")
def prefix_transformer_iwslt_de_en(args):
    args.encoder_prefix_length = getattr(args, "encoder_prefix_length", 20)
    args.decoder_prefix_length = getattr(args, "decoder_prefix_length", 20)
    args.ignore_prefix_size = args.decoder_prefix_length
    args.encoder_prefix_init = getattr(args, "encoder_prefix_init", "from-vocab")
    transformer_mbart_large(args)
