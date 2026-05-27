import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.layers import trunc_normal_
from timm.models import register_model
from Models.modeling_finetune import NeuralTransformer
from Models.norm_ema_quantizer import NormEMAVectorQuantizer


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    @staticmethod
    def _dot_simililarity(x, y):
        return torch.matmul(x, y.T)

    def _cosine_simililarity(self, x, y):
        return self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0)).squeeze()

    def _get_correlated_mask(self, batch_size, device):
        N = 2 * batch_size
        eye = torch.eye(N, device=device)
        mask = 1 - (eye + torch.roll(eye, shifts=batch_size, dims=1) + torch.roll(eye, shifts=-batch_size, dims=1))
        return mask.type(torch.bool)

    def forward(self, zis, zjs):
        device = zis.device
        batch_size = zis.size(0)

        zis = zis.view(batch_size, -1)
        zjs = zjs.view(batch_size, -1)

        representations = torch.cat([zjs, zis], dim=0).to(device)
        similarity_matrix = self.similarity_function(representations, representations)

        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask = self._get_correlated_mask(batch_size, device)
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)
        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)


class Tokenizer(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_embed_t=4096,
                 n_embed_f=4096,
                 embed_dim=32,
                 decay=0.99,
                 quantize_kmeans_init=True,
                 decoder_out_dim=200,
                 smooth_l1_loss=False,
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        if decoder_config['in_chans'] != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config['in_chans']} to {embed_dim}")
            decoder_config['in_chans'] = embed_dim

        print('Final encoder config', encoder_config)
        self.encoder = NeuralTransformer(**encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder_t = NeuralTransformer(**decoder_config)
        self.decoder_f = NeuralTransformer(**decoder_config)

        self.quantize_t = NormEMAVectorQuantizer(
            n_embed=n_embed_t, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init,
            decay=decay,
        )

        self.quantize_f = NormEMAVectorQuantizer(
            n_embed=n_embed_f, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init,
            decay=decay,
        )

        self.patch_size = encoder_config['patch_size']
        self.decoder_out_dim = decoder_out_dim

        self.encode_task_layer_t = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim)
        )

        self.encode_task_layer_f = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim)
        )

        self.decode_task_layer_temp = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.decode_task_layer_xrec = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.decode_task_layer_angle = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )

        self.kwargs = kwargs

        self.encode_task_layer_t.apply(self._init_weights)
        self.encode_task_layer_f.apply(self._init_weights)
        self.decode_task_layer_temp.apply(self._init_weights)
        self.decode_task_layer_xrec.apply(self._init_weights)
        self.decode_task_layer_angle.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self._infonce_loss = None

    def get_infonce_loss(self):
        if self._infonce_loss is None:
            self._infonce_loss = NTXentLoss(
                temperature=0.1,
                use_cosine_similarity=True
            )
        return self._infonce_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed',
                'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed'}

    @property
    def device(self):
        return next(self.parameters()).device

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, input_chans=list(range(20)), **kwargs):
        t_quantize, t_embed_ind, t_emb_loss, f_quantize, f_embed_ind, f_emb_loss, _ = (
            self.encode(data, input_chans=input_chans))
        output = {}
        output['token_t'] = t_embed_ind.view(data.shape[0], -1)
        output['input_img'] = data
        output['quantize_t'] = rearrange(t_quantize, 'b d a c -> b (a c) d')
        output['token_f'] = f_embed_ind.view(data.shape[0], -1)
        output['quantize_f'] = rearrange(f_quantize, 'b d a c -> b (a c) d')
        return output

    def encode(self, x, input_chans=list(range(20))):
        batch_size, n, a, t = x.shape
        encoder_features = self.encoder(x, input_chans, return_patch_tokens=True)
        if torch.is_grad_enabled():
            encoder_features_half1= self.encoder(x[:,:,:a//2,:], input_chans, return_patch_tokens = True)
            encoder_features_half2= self.encoder(x[:,:,a//2:,:], input_chans, return_patch_tokens = True)

            cl_loss = self.get_infonce_loss()(encoder_features_half1, encoder_features_half2)
        else:
            cl_loss = torch.tensor(0.0, device=x.device)
        with torch.amp.autocast('cuda', enabled=False):
            to_quantizer_features_t = self.encode_task_layer_t(
                encoder_features.type_as(self.encode_task_layer_t[-1].weight))
            to_quantizer_features_f = self.encode_task_layer_f(
                encoder_features.type_as(self.encode_task_layer_f[-1].weight))

        N = to_quantizer_features_t.shape[1]
        h, w = n, N // n

        to_quantizer_features_t = rearrange(to_quantizer_features_t,
                                            'b (h w) c -> b c h w', h=h, w=w)
        to_quantizer_features_f = rearrange(to_quantizer_features_f,
                                            'b (h w) c -> b c h w', h=h, w=w)
        t_quantize, t_emb_loss, t_embed_ind = self.quantize_t(to_quantizer_features_t)
        f_quantize, f_emb_loss, f_embed_ind = self.quantize_f(to_quantizer_features_f)

        return t_quantize, t_embed_ind, t_emb_loss, f_quantize, f_embed_ind, f_emb_loss, cl_loss

    def decode_f(self, quantize, input_chans=list(range(20)), **kwargs):
        decoder_features = self.decoder_f(quantize, input_chans, return_patch_tokens=True)
        rec = self.decode_task_layer_xrec(decoder_features)
        rec_angle = self.decode_task_layer_angle(decoder_features)
        return rec, rec_angle

    def decode_t(self, quantize, input_chans=list(range(20)), **kwargs):
        decoder_features = self.decoder_t(quantize, input_chans, return_patch_tokens=True)
        recx_raw = self.decode_task_layer_temp(decoder_features)
        return recx_raw

    def get_codebook_indices(self, x, input_chans=list(range(20)), **kwargs):
        output = self.get_tokens(x, input_chans, **kwargs)
        return output['token_t'], output['token_f']

    def calculate_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c')
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / std
        return x

    def forward(self, x, input_chans=list(range(20))):
        raw_siganls = x
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        amplitude = self.std_norm(amplitude)
        angle = torch.angle(x_fft)
        angle = self.std_norm(angle)

        t_quantize, t_embed_ind, t_emb_loss, f_quantize, f_embed_ind, f_emb_loss, cl_loss \
            = self.encode(x, input_chans=input_chans)

        recx_raw = self.decode_t(t_quantize, input_chans)
        rec_raw_loss = self.calculate_rec_loss(recx_raw, raw_siganls)

        recx_spec, recx_angle = self.decode_f(f_quantize, input_chans)
        rec_spec_loss = self.calculate_rec_loss(recx_spec, amplitude)
        rec_angle_loss = self.calculate_rec_loss(recx_angle, angle)
        losses = [t_emb_loss, f_emb_loss, rec_raw_loss, rec_spec_loss, rec_angle_loss, cl_loss]
        device = t_emb_loss.device
        loss = sum(l.to(device) for l in losses)
        log = {}
        split = "train" if self.training else "val"
        log[f'{split}/quant_t_loss'] = t_emb_loss.detach().mean()
        log[f'{split}/quant_f_loss'] = f_emb_loss.detach().mean()
        log[f'{split}/rec_raw_loss'] = rec_raw_loss.detach().mean()
        log[f'{split}/rec_spec_loss'] = rec_spec_loss.detach().mean()
        log[f'{split}/rec_angle_loss'] = rec_angle_loss.detach().mean()
        log[f'{split}/cl_loss'] = cl_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, log


def get_model_default_params():
    return dict(EEG_size=6000, patch_size=200, in_chans=1, num_classes=1000, embed_dim=200,
                depth=12, num_heads=10, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0., use_abs_pos_emb=True, use_rel_pos_bias=False,
                use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)


@register_model
def tfdual_vq(pretrained=False, pretrained_weight=None, as_tokenzer=False,
                                        EEG_size=6000, n_code_t=4096, n_code_f=4096, code_dim=32, **kwargs):
    encoder_config, decoder_config = get_model_default_params(), get_model_default_params()
    encoder_config['EEG_size'] = EEG_size
    encoder_config['num_classes'] = 0
    decoder_config['EEG_size'] = EEG_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = code_dim
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 3
    decoder_out_dim = 200

    model = Tokenizer(encoder_config, decoder_config, n_code_t, n_code_f, code_dim,
                  decoder_out_dim=decoder_out_dim, **kwargs)

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None

        if pretrained_weight.startswith('https'):
            weights = torch.hub.load_state_dict_from_url(pretrained_weight, map_location='cpu', check_hash=True)
        else:
            weights = torch.load(pretrained_weight, map_location='cpu', weights_only=False)

        if 'model' in weights:
            weights = weights['model']
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())

        for k in keys:
            if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
                del weights[k]
        model.load_state_dict(weights)
    return model