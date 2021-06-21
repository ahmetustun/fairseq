# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from typing import Optional

from omegaconf import DictConfig
from argparse import Namespace

import torch
from torch import nn
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
    transformer_wmt_en_de_big,
    TransformerEncoder, TransformerDecoder, transformer_iwslt_de_en, transformer_mbart_large)

logger = logging.getLogger(__name__)

@register_model("prompt_transformer")
class PromptTransformer(TransformerModel):
    """
    See "The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)"
    """

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(PromptTransformer, PromptTransformer).add_args(parser)
        # Prompt tuning
        parser.add_argument('--encoder-prompt-length', type=int, metavar='N', default=200,
                            help='encoder prompt embedding length')
        parser.add_argument('--encoder-prompt-init', type=str, metavar='N', default='from-vocab',
                            help='encoder prompt embedding init method [from-vocab, uniform] ')
        parser.add_argument('--decoder-prompt-length', type=int, metavar='N', default=None,
                            help='encoder prompt embedding length')
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.args = args
        for n, p in self.named_parameters():
            if 'prompt' not in n:
                p.requires_grad = False

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        prompt_transformer(args)

        if args.encoder_prompt_length != 0:
            encoder_prompts = cls.build_prompt(args.encoder_prefix_length, args.encoder_embed_dim)
        else:
            encoder_prompts = None

        if args.decoder_prompt_length != 0:
            decoder_prompts = cls.build_prompt(args.encoder_prefix_length, args.encoder_embed_dim)
        else:
            decoder_prompts = None

        cls.encoder_prompts = encoder_prompts
        cls.decoder_prompts = decoder_prompts

        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return PromptTransformerEncoder(args, src_dict, embed_tokens, cls.encoder_prompts)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False))

    @classmethod
    def build_prompt(cls, prompt_length, embedding_dim):
        prompts = PromptEmbeddings(prompt_length, embedding_dim)
        return prompts

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None):

        # Do not enforce the key match due to the prompt params
        strict = False

        # initialization of prompt tokens
        state = super().load_state_dict(state_dict, strict)

        logger.info(f'missing keys: {state.missing_keys}')
        logger.info(f'missing keys: {state.unexpected_keys}')

        if self.args.encoder_prompt_init == 'from-vocab':
            input_embeddings = state_dict['encoder.embed_tokens.weight']
            sample_tokens_idx = random.sample(range(0, input_embeddings.shape[0]), self.args.encoder_prompt_length)
            prompt_dict = {'weight': input_embeddings[sample_tokens_idx]}
            self.encoder.prompts.load_state_dict(prompt_dict, True)
            logger.info(f'encoder.prompts.weight is initialized from vocabulary')
        else:
            logger.info(f'encoder.prompts.weight is uniformly initialized')

        return state

class PromptEmbeddings(nn.Module):

    def __init__(self, prompt_length, embedding_dim):
        super(PromptEmbeddings, self).__init__()
        self.prompt_length = prompt_length
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.zeros(prompt_length, embedding_dim))
        self.init_weights()

    def init_weights(self, range=None):
        self.weight.data.normal_(mean=0, std=self.embedding_dim ** -0.5)

    def get_prompt_length(self):
        return self.prompt_length

    def forward(self, bsz):
        return self.prompts.unsqueeze(0).repeat(bsz,1,1)


class PromptTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, encoder_prompts=None):
        super().__init__(args, dictionary, embed_tokens)

        self.prompts = encoder_prompts

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None,
            encoder_prompts: Optional[torch.Tensor] = None):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding

        # adding prompts: concat in sequence dim
        bsz = src_tokens.shape[0]
        embed = torch.cat((self.encoder_prompts(bsz), embed), 1)

        # dummy prompt tokens for positional embeddings
        dummy_promts = torch.zeros(bsz, self.encoder_prompts.get_prompt_length())

        if self.embed_positions is not None:
            x = embed + self.embed_positions(torch.cat((dummy_promts, src_tokens), 1))
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None):
        # compute padding mask with dummy prompt tokens for positional embeddings
        bzs = bsz = src_tokens.shape[0]
        dummy_promts = torch.zeros(bsz, self.encoder_prompts.get_prompt_length())
        encoder_padding_mask = torch.cat((dummy_promts,src_tokens),1).eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


@register_model_architecture("prompt_transformer", "prompt_transformer")
def prompt_transformer(args):
    args.encoder_prefix_length = getattr(args, "encoder_prefix_length", 200)
    args.decoder_prefix_length = getattr(args, "decoder_prefix_length", 0)
    args.encoder_prompt_init = getattr(args, "encoder_prompt_init", "from-vocab")
    base_architecture(args)


@register_model_architecture("prompt_transformer", "prompt_transformer_wmt_en_de_big")
def prompt_transformer_wmt_en_de_big(args):
    args.encoder_prefix_length = getattr(args, "encoder_prefix_length", 200)
    args.decoder_prefix_length = getattr(args, "decoder_prefix_length", 0)
    transformer_wmt_en_de_big(args)


@register_model_architecture("prompt_transformer", "prompt_transformer_iwslt_de_en")
def prompt_transformer_iwslt_de_en(args):
    args.encoder_prefix_length = getattr(args, "encoder_prefix_length", 200)
    args.decoder_prefix_length = getattr(args, "decoder_prefix_length", 0)
    transformer_iwslt_de_en(args)


@register_model_architecture("prompt_transformer", "prompt_transformer_mbart_large")
def prompt_transformer_iwslt_de_en(args):
    args.encoder_prefix_length = getattr(args, "encoder_prefix_length", 200)
    args.decoder_prefix_length = getattr(args, "decoder_prefix_length", 0)
    transformer_mbart_large(args)
