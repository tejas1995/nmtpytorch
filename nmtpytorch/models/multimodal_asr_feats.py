# -*- coding: utf-8 -*-
import logging

from torch import nn
import torch.nn.functional as F

from ..layers import ImageEncoder, BiLSTMp, ConditionalMMFeatFusionDecoder
from ..datasets import MultimodalDataset
from .nmt import NMT
from .asr import ASR

logger = logging.getLogger('nmtpytorch')


class MultimodalASRFeats(ASR):
    """An end-to-end sequence-to-sequence ASR model with feature fusion in the decoder
    """
    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'cnn_type': 'resnet50',     # A variant of VGG or ResNet
            'cnn_layer': 'res5c_relu',  # From where to extract features
            'cnn_pretrained': True,     # Should we use pretrained imagenet weights
            'cnn_finetune': None,       # Should we finetune part or all of CNN
            'pool': None,               # ('Avg|Max', kernel_size, stride_size)
            'dropout_img': 0.,          # a 2d dropout over conv features
            'l2_norm': False,           # L2 normalize features
            'l2_norm_dim': -1,           # Which dimension to L2 normalize
            'resize': 256,              # resize width, height for images
            'crop': 224,                # center crop size after resize
            'feat_fusion': 'early_concat',
            'aux_dim': 2048,
            'aux_proj_dim': 256,
            'aux_activ': 'tanh',
            'feat_activ': 'tanh',
        })

    def __init__(self, opts):
        super().__init__(opts)

        assert self.opts.model['cnn_layer'] not in ('avgpool', 'fc', 'pool'), \
            "{} given for 'cnn_layer' but it should be a conv layer.".format(self.opts.model['cnn_layer'])

    def reset_parameters(self):
        """Initializes learnable weights with kaiming normal."""
        for name, param in self.named_parameters():
            if (param.requires_grad and 'bias' not in name and
                    not name.startswith('cnn')):
                logger.info('  Initializing weights for {}'.format(name))
                nn.init.kaiming_normal_(param.data)

        if self.opts.model['lstm_bias_zero'] or \
                self.opts.model['lstm_forget_bias']:
            for name, param in self.speech_enc.named_parameters():
                if 'bias_hh' in name or 'bias_ih' in name:
                    # Reset bias to 0
                    param.data.fill_(0.0)
                    if self.opts.model['lstm_forget_bias']:
                        # Reset forget gate bias of LSTMs to 1
                        # the tensor organized as: inp,forg,cell,out
                        n = param.numel()
                        param[n // 4: n // 2].data.fill_(1.0)


    def setup(self, is_train=True):
        logger.info('Loading CNN')
        cnn_encoder = ImageEncoder(
            cnn_type=self.opts.model['cnn_type'],
            pretrained=self.opts.model['cnn_pretrained'])

        # Set truncation point
        cnn_encoder.setup(layer=self.opts.model['cnn_layer'],
                          dropout=self.opts.model['dropout_img'],
                          pool=self.opts.model['pool'])

        # By default the CNN is not tuneable
        if self.opts.model['cnn_finetune'] is not None:
            assert not self.opts.model['l2_norm'], \
                "finetuning and l2 norm does not work together."
            cnn_encoder.set_requires_grad(
                value=True, layers=self.opts.model['cnn_finetune'])

        # Number of channels defines the spatial vector dim for us
        self.ctx_sizes['image'] = cnn_encoder.get_output_shape()[1]

        # Finally set the CNN as a submodule
        self.cnn = cnn_encoder.get()

        # Nicely printed table of summary for the CNN
        logger.info(cnn_encoder)

        ########################
        # Create Speech Encoder
        ########################
        self.speech_enc = BiLSTMp(
            input_size=self.opts.model['feat_dim'],
            hidden_size=self.opts.model['enc_dim'],
            proj_size=self.opts.model['proj_dim'],
            proj_activ=self.opts.model['proj_activ'],
            dropout=self.opts.model['dropout'],
            layers=self.opts.model['enc_layers'])


        # Create Decoder
        self.dec = ConditionalMMFeatFusionDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.src),
            tied_emb=self.opts.model['tied_dec_embs'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout'],
            feat_fusion=self.opts.model['feat_fusion'],
            aux_dim=self.opts.model['aux_dim'],
            aux_proj_dim=self.opts.model['aux_proj_dim'],
            aux_activ=self.opts.model['aux_activ'])


    def load_data(self, split, batch_size, mode='train', dump_attn=False):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data[split + '_set'],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model.get('max_len', None),
            warmup=(split != 'train'),
            resize=self.opts.model['resize'],
            crop=self.opts.model['crop'],
            dump_attn=dump_attn)
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        # Get features into (n,c,w*h) and then (w*h,n,c)
        # feats = self.cnn(batch['image'].permute(1, 0, 2, 3))
        # feats = feats.view((*feats.shape[:2], -1)).permute(2, 0, 1)
        # if self.opts.model['l2_norm']:
        #     feats = F.normalize(
        #         feats, dim=self.opts.model['l2_norm_dim']).detach()

        # x = batch[self.src]
        # if self.opts.model['feat_transform']:
        #     x = self.feat_transform(x)
        # return {
        #     'image': (feats, None),
        #     str(self.src): self.speech_enc(x),
        # }

        d = {str(self.src): self.speech_enc(batch[self.src], aux=batch['feats'])}

        if 'feats' in batch:
            d['feats'] = (batch['feats'], None)
      
        return d
