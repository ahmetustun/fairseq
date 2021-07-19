import logging

from fairseq.modules.transformer_layer import TransformerEncoderLayer
from fairseq.modules.transformer_layer import TransformerDecoderLayer
from fairseq.modules.adapter_layer import AdapterLayer
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
    transformer_wmt_en_de_big,
    transformer_iwslt_de_en,
    transformer_mbart_large, TransformerDecoder, TransformerEncoder)
from fairseq.models import register_model, register_model_architecture
from torch import nn


logger = logging.getLogger(__name__)


class AdapterTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, *args_, **kwargs):
        super().__init__(args, *args_, **kwargs)

        decoder_only = getattr(args, "decoder_only", False)
        pfeiffer = getattr(args, "pfeiffer", False)
        init = getattr(args, "adapter_init", 'small')
        self.skip = decoder_only
        if not self.skip:
            adapter_dim = getattr(args, 'adapter_dim', 64)

            self.adapter = AdapterLayer(args.encoder_embed_dim, adapter_dim, pfeiffer=pfeiffer,
                                  init=init)

            self.return_prenorm = pfeiffer

    def forward(self, x, *args, task=None, **kwargs):
        x = super().forward(x, *args, **kwargs)

        if not self.skip:
            if self.return_prenorm:
                y = self.adapter(self.final_layer_norm(x))
                x = self.final_layer_norm(y + x)
            else:
                x = self.adapter(x)

        return x


class AdapterTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args, *args_, **kwargs):
        super().__init__(args, *args_, **kwargs)
        encoder_only = getattr(args, "encoder_only", False)
        pfeiffer = getattr(args, "pfeiffer", False)
        init = getattr(args, "adapter_init", 'small')
        skip = getattr(args, "ignore_dec_layers", [])
        self.skip = encoder_only
        self.adapt_encoder_out = getattr(args, "adapt_encoder_out", False)
        if not self.skip:
            adapter_dim = getattr(args, 'adapter_dim', 64)

            self.adapter = AdapterLayer(args.decoder_embed_dim, adapter_dim, pfeiffer=pfeiffer,
                                  init=init)

            if self.adapt_encoder_out:
                self.enc_adapters = AdapterLayer(args.decoder_embed_dim, adapter_dim, pfeiffer=pfeiffer,
                                      init=init)

            self.return_prenorm = pfeiffer

    def forward(self, x, encoder_out, *args, task=None, **kwargs):
        if self.adapt_encoder_out:
            encoder_out = self.enc_adapters(encoder_out)
        x, *extra = super().forward(x, encoder_out, *args, **kwargs)

        if not self.skip:
            if self.return_prenorm:
                y = self.adapter(self.final_layer_norm(x))
                x = self.final_layer_norm(y + x)
            else:
                x = self.adapter(x)

        return (x, *extra)


class AdapterTransformerEncoder(TransformerEncoder):
    def build_encoder_layer(self, *args, **kwargs):
        return AdapterTransformerEncoderLayer(*args, **kwargs)


class AdapterTransformerDecoder(TransformerDecoder):
    def build_decoder_layer(self, *args, **kwargs):
        return AdapterTransformerDecoderLayer(*args, **kwargs)



@register_model('adapter_transformer')
class AdapterTransformerModel(TransformerModel):
    """
    Only overriding build_encoder and build_decoder methods.
    """
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        super().add_args(parser)
        parser.add_argument('--adapter-dim', type=int, default=64)
        parser.add_argument('--pfeiffer', action='store_true')
        parser.add_argument('--encoder-only', action='store_true')
        parser.add_argument('--decoder-only', action='store_true')
        parser.add_argument('--adapt-encoder-out', action='store_true')
        parser.add_argument('--adapter-init', type=str, default='small')

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        for name, parameter in self.named_parameters():
            #if "decoder.layers.5" not in name:
            if "adapters" not in name:
                parameter.requires_grad = False

    @classmethod
    def build_encoder(cls, *args, **kwargs):
        return AdapterTransformerEncoder(*args, **kwargs)

    @classmethod
    def build_decoder(cls, *args, **kwargs):
        return AdapterTransformerDecoder(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=False, args=None):
        """
        Some hacks to load TransformerModel checkpoints into
        AdapterModel.
        """
        self.upgrade_state_dict(state_dict)

        status = super().load_state_dict(state_dict, strict=False)

        if status.missing_keys:
            logger.info("Missing keys detected")

        if status.unexpected_keys:
            logger.info("Unexpected keys found")


@register_model_architecture('adapter_transformer', 'adapter_transformer')
def adapter_transformer(args):
    base_architecture(args)

@register_model_architecture('adapter_transformer', 'adapter_transformer_iwslt_de_en')
def adapter_transformer_iwslt_de_en(args):
    transformer_iwslt_de_en(args)

@register_model_architecture('adapter_transformer', 'adapter_transformer_wmt_en_de_big')
def adapter_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)

@register_model_architecture('adapter_transformer', 'adapter_transformer_mbart_large')
def adapter_transformer_mbart_large(args):
    transformer_mbart_large(args)
