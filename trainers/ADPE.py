import copy
import os.path as osp

import math
from functools import reduce
from operator import mul
from collections import OrderedDict
import operator

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Dropout
from torch.cuda.amp import GradScaler, autocast
from einops import rearrange

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import MetricMeter, AverageMeter

from clip import clip
from clip.model import LayerNorm
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.utils import (
    load_clip_to_cpu,
    CUSTOM_TEMPLATES,
    IMAGENET_TEMPLATES_SELECT,
)

import time
import numpy as np
import datetime
import csv
import json
from pathlib import Path
from tqdm import tqdm
from utils.adpe_ablation import make_ablation_masks
from utils.adpe_descriptions import load_gpt_descriptions, _is_imagenet_dataset

_tokenizer = _Tokenizer()


class VisualPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model, num_prompts):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = clip_model.visual.ln_post.weight.shape[0]
        self.deep = cfg.MODEL.ADPE.DEEP
        self.deep_layers = 12
        self.dtype = clip_model.dtype
        self.init_prompt(clip_model.visual.conv1.kernel_size)
        self.in_dim = clip_model.ln_final.weight.shape[0]
        pooling_cfg = cfg.MODEL.ADPE.POOLING
        norm_layer = LayerNorm if pooling_cfg.NORM == 'layernorm' else lambda x: x / x.norm(dim=-1, keepdim=True)
        self.poolings = [
            AttentionPooling(dim=self.in_dim, norm_layer=norm_layer)
            for _ in range(self.num_prompts + 1)
        ]
        self.poolings = nn.ModuleList(self.poolings)

    def init_prompt(self, patch_size):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.prompt_dim))
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.num_prompts, self.prompt_dim).type(self.dtype))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        if self.deep:
            total_d_layer = self.deep_layers - 1
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                total_d_layer, self.num_prompts, self.prompt_dim).type(self.dtype))
            nn.init.zeros_(self.deep_prompt_embeddings.data)
        else:
            self.deep_prompt_embeddings = None
    def forward(self):
        return self.prompt_embeddings, self.deep_prompt_embeddings

class VisualEncoder(nn.Module):
    def __init__(self, cfg, clip_model, num_prompts):
        super().__init__()
        self.input_resolution = clip_model.visual.input_resolution
        self.output_dim = clip_model.visual.output_dim
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre =  clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj
        self.deep = cfg.MODEL.ADPE.DEEP
        self.prompt_dropout = Dropout(cfg.MODEL.ADPE.DROPOUT)
        self.num_prompts = num_prompts
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

    def forward_deep_prompt(self, embedding_output, deep_prompt_embeddings):
        hidden_states = None
        B = embedding_output.shape[1]
        num_layers = len(self.transformer.resblocks)
        for i in range(num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
            else:
                if i <= deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(
                        deep_prompt_embeddings[i-1].expand(B, -1, -1))
                    hidden_states = hidden_states.permute(1, 0, 2)
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb+hidden_states[:, 1:(1+self.num_prompts), :],
                        hidden_states[:, (1+self.num_prompts):, :]
                    ), dim=1)
                    hidden_states = hidden_states.permute(1, 0, 2)
                hidden_states = self.transformer.resblocks[i](hidden_states)

        return hidden_states

    def prepare_token(self, x, prompt_embeddings):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        B = x.shape[0]
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(prompt_embeddings.expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        return x

    def forward(self, x, prompt_embeddings, deep_prompt_embeddings, return_img_tokens=False):
        x = self.prepare_token(x, prompt_embeddings)
        if self.deep:
            x = self.forward_deep_prompt(x, deep_prompt_embeddings)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)
        cls_feature = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            cls_feature = cls_feature @ self.proj
        prompts = self.ln_post(x[:, 1:1+self.num_prompts, :])
        if self.proj is not None:
            prompts = prompts @ self.proj
        return cls_feature, prompts

class CrossAttention(nn.Module):
    def __init__(self, dim=512, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.init_weights()

    def init_weights(self):
        for child_name, child_module in self.named_children():
            if isinstance(child_module, nn.Linear):
                nn.init.trunc_normal_(child_module.weight, std=0.02)
                if child_module.bias is not None:
                    nn.init.zeros_(child_module.bias)

    def forward(self, x, attn_mask):
        B, N, C = x.shape
        k = self.k(x[:, 1:, :])
        q = self.q(x[:, 0:1, :])
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + (1 - attn_mask[:, 1:]).unsqueeze(1) * torch.finfo(attn.dtype).min
        attn = attn.softmax(dim=-1)
        x = attn @ x[:, 1:, :]
        return x

class AttentionPooling(nn.Module):
    def __init__(self, dim, norm_layer=LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim) if isinstance(norm_layer, type) else norm_layer
        self.attn = CrossAttention(dim)

    def forward(self, img_token, text_embeddings, attn_mask):
        B = img_token.shape[0]
        num_classes = text_embeddings.shape[1]
        x = torch.cat([img_token.repeat(1, num_classes, 1, 1), text_embeddings.repeat(B, 1, 1, 1)],dim=-2)
        x = rearrange(x, 'b n_c n d -> (b n_c) n d')
        attn_mask = attn_mask.repeat(B, 1, 1)
        attn_mask = rearrange(attn_mask, 'b n_c n -> (b n_c) n')
        x = self.attn(self.norm(x), attn_mask)
        x = rearrange(x, '(b n_c) n d -> b n_c n d', b=B)
        return x.squeeze(2)

def make_description_batch(descriptor_text_prompt: dict, max_len_quantile=1.0):
    des_len = np.array([v.shape[0] for v in descriptor_text_prompt.values()])
    if max_len_quantile < 1.0:
        max_len = int(np.quantile(des_len, max_len_quantile))
    else:
        max_len = int(np.max(des_len))
    num_classes = len(descriptor_text_prompt)
    batched_descriptor_text_prompt = []
    batched_attn_mask = []
    for idx, (cls_name, text_prompt) in enumerate(descriptor_text_prompt.items()):
        curr_len = text_prompt.shape[0]
        if curr_len > max_len:
            curr_len = max_len
        attn_mask = torch.ones(curr_len)
        batched_descriptor_text_prompt.append(torch.cat([text_prompt[:max_len], torch.zeros(max_len-curr_len, text_prompt.shape[1], dtype=torch.long)]))
        attn_mask = torch.cat([attn_mask, torch.zeros(max_len-curr_len, dtype=torch.long)])
        batched_attn_mask.append(attn_mask)
    batched_descriptor_text_prompt = torch.stack(batched_descriptor_text_prompt)
    batched_attn_mask = torch.stack(batched_attn_mask)

    return batched_descriptor_text_prompt, batched_attn_mask

class TextPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, lookup_aliases=None):
        super().__init__()
        dataset_name = cfg.DATASET.NAME
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        prompts = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompts).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        print(f'Initial context: "{ctx_init}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.text_prompts, self.lookup_classnames = load_gpt_descriptions(
            dataset_name, classnames, cfg.MODEL.ADPE, IMAGENET_TEMPLATES_SELECT, lookup_aliases
        )
        self.num_prompts = len(self.text_prompts)-1

    def forward(self, token_prefix, token_suffix, init=False):
        if init:
            ctx = self.ctx.clone().detach().cuda()
        else:
            ctx = self.ctx
        B = token_prefix.shape[0]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(B, -1, -1)
        prompts = torch.cat(
            [
                token_prefix,
                ctx,
                token_suffix,
            ],
            dim=1,
        )
        return prompts

class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model, classnames, text_prompt_learner, poolings):
        super().__init__()
        clip_model.cuda()
        self.text_prompt_learner = text_prompt_learner
        text_prompts = text_prompt_learner.text_prompts
        self.text_prompts = {description_key: {class_key: clip.tokenize(descriptions) for class_key, descriptions in v.items()} for description_key, v in text_prompts.items()}
        self.poolings = poolings
        self.max_len_quantile = cfg.MODEL.ADPE.MAX_LEN_QUANTILE
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.classnames = classnames
        self.descriptor_batched_attn_masks = {}
        self.descriptor_text_features = {}
        self.descriptor_token_preffix = {}
        self.descriptor_token_suffix = {}
        self.descriptor_text_prompt = {}
        self.n_ctx = 4
        self.preextract_frozen()

    def encode_text(self, text, token_embed=None, embed_only=False):
        if token_embed is None:
            token_embed = self.token_embedding(text).type(self.dtype)
        if embed_only:
            return token_embed
        x = token_embed + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x, token_embed

    def sample_text_features(self, text_features, frozen_text_features, attn_mask):
        non_zero_indices = attn_mask.nonzero(as_tuple=True)
        sampled_indices = torch.zeros(attn_mask.shape[0], dtype=torch.long).to(attn_mask.device)
        for row_idx in range(attn_mask.shape[0]):
            row_non_zero_indices = non_zero_indices[1][non_zero_indices[0] == row_idx]
            sampled_index = row_non_zero_indices[torch.randint(0, len(row_non_zero_indices), (1,))]
            sampled_indices[row_idx] = sampled_index
        sampled_indices = sampled_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, text_features.shape[-1])
        sampled_features = torch.gather(text_features, 1, sampled_indices)
        sampled_frozen_features = torch.gather(frozen_text_features, 1, sampled_indices)
        sampled_features = sampled_features.squeeze(1)
        sampled_frozen_features = sampled_frozen_features.squeeze(1)
        return sampled_features, sampled_frozen_features

    @torch.no_grad()
    def preextract_frozen(self):
        for des_i, (description_key, description_prompts) in enumerate(self.text_prompts.items()):
            if description_key not in self.descriptor_batched_attn_masks:
                batched_descriptor_text_prompt, batched_attn_mask = make_description_batch(description_prompts, self.max_len_quantile)
                rearranged_text_prompts = rearrange(batched_descriptor_text_prompt, 'n_c n d -> (n_c n) d')
                text_features, token_embeds = self.encode_text(rearranged_text_prompts.cuda())
                n_c = len(self.classnames)
                if len(description_prompts) != n_c:
                    raise RuntimeError(
                        "[ADPE TextEncoder] Prompt dictionary/class count mismatch.\n"
                        f"  descriptor={description_key}\n"
                        f"  prompt class keys={len(description_prompts)}\n"
                        f"  dataset classes={n_c}\n"
                        "  Duplicate class names may have collapsed in a dict."
                    )
                if text_features.shape[0] % n_c != 0:
                    raise RuntimeError(
                        "[ADPE TextEncoder] Text feature count is not divisible "
                        "by class count.\n"
                        f"  descriptor={description_key}\n"
                        f"  text_features.shape={tuple(text_features.shape)}\n"
                        f"  dataset classes={n_c}"
                    )
                text_features = rearrange(text_features, '(n_c n) d -> n_c n d', n_c=n_c)
                token_embeds = rearrange(token_embeds, '(n_c n) v d -> n_c n v d', n_c=n_c)
                token_preffix = token_embeds[:,:,:1,:]
                token_suffix = token_embeds[:,:,1+self.n_ctx:,:]
                self.descriptor_batched_attn_masks[description_key] = batched_attn_mask
                self.descriptor_token_preffix[description_key] = token_preffix
                self.descriptor_token_suffix[description_key] = token_suffix
                self.descriptor_text_features[description_key] = text_features
                self.descriptor_text_prompt[description_key] = batched_descriptor_text_prompt

        self.frozen_text_features = copy.deepcopy(self.descriptor_text_features)

    @torch.no_grad()
    def update_text_features(self):
        for des_i, (description_key, description_prompts) in enumerate(self.text_prompts.items()):
            token_preffix = self.descriptor_token_preffix[description_key]
            token_suffix = self.descriptor_token_suffix[description_key]
            rearranged_text_prompts = self.descriptor_text_prompt[description_key]
            token_preffix = rearrange(token_preffix, 'n_c n v d -> (n_c n) v d')
            token_suffix = rearrange(token_suffix, 'n_c n v d -> (n_c n) v d')
            rearranged_text_prompts = rearrange(rearranged_text_prompts, 'n_c n d -> (n_c n) d')
            prompts = self.text_prompt_learner(token_preffix, token_suffix)
            text_features, _ = self.encode_text(rearranged_text_prompts, prompts)
            text_features = rearrange(text_features, '(n_c n) d -> n_c n d', n_c=len(self.classnames))
            self.descriptor_text_features[description_key] = text_features

    def forward(self, prompt_img_features, cls_img_feature, alternating_epoch=0, sampled_classnames=None, old_indices=None):
        device = prompt_img_features.device
        B = prompt_img_features.shape[0]
        if sampled_classnames is not None:
            text_prompts = {description_key: {class_key: v[class_key] for class_key in sampled_classnames} for description_key, v in self.text_prompts.items()}
            n_c = len(sampled_classnames)
        else:
            text_prompts = self.text_prompts
            n_c = len(self.classnames)
        if self.training and alternating_epoch == 0 and not self.text_prompt_learner.training:
            self.update_text_features()
        text_embeddings = []
        sampled_text_features = []
        sampled_frozen_text_features = []
        for des_i, (description_key, description_prompts) in enumerate(text_prompts.items()):
            if self.text_prompt_learner.training:
                batched_attn_mask = self.descriptor_batched_attn_masks[description_key]
                token_preffix = self.descriptor_token_preffix[description_key]
                token_suffix = self.descriptor_token_suffix[description_key]
                rearranged_text_prompts = self.descriptor_text_prompt[description_key]
                frozen_text_features = self.frozen_text_features[description_key]
                if old_indices is not None:
                    batched_attn_mask = batched_attn_mask[old_indices]
                    token_preffix = token_preffix[old_indices]
                    token_suffix = token_suffix[old_indices]
                    rearranged_text_prompts = rearranged_text_prompts[old_indices]
                    frozen_text_features = frozen_text_features[old_indices]
                token_preffix = rearrange(token_preffix, 'n_c n v d -> (n_c n) v d')
                token_suffix = rearrange(token_suffix, 'n_c n v d -> (n_c n) v d')
                rearranged_text_prompts = rearrange(rearranged_text_prompts, 'n_c n d -> (n_c n) d')
                prompts = self.text_prompt_learner(token_preffix, token_suffix)
                text_features, _ = self.encode_text(rearranged_text_prompts, prompts)
                text_features = rearrange(text_features, '(n_c n) d -> n_c n d', n_c=n_c)
                sampled_text_feature, sampled_frozen_text_feature = self.sample_text_features(text_features, frozen_text_features, batched_attn_mask.cuda())
                sampled_text_features.append(sampled_text_feature)
                sampled_frozen_text_features.append(sampled_frozen_text_feature)
            else:
                text_features = self.descriptor_text_features[description_key].detach()
                batched_attn_mask = self.descriptor_batched_attn_masks[description_key]
            batched_attn_mask = torch.cat([torch.ones(n_c, 1, dtype=torch.long), batched_attn_mask], dim=1).to(device)
            pooling_layer = self.poolings[des_i]
            if description_key != 'cls':
                prompt_img_feature = prompt_img_features[:, des_i:des_i+1, :]
                descriptor_text_features = pooling_layer(prompt_img_feature.unsqueeze(1), text_features.unsqueeze(0), batched_attn_mask.unsqueeze(0))
                descriptor_text_features = descriptor_text_features / descriptor_text_features.norm(dim=-1, keepdim=True)
                text_embeddings.append(descriptor_text_features)
            else:
                descriptor_text_features = pooling_layer(cls_img_feature.unsqueeze(1), text_features.unsqueeze(0), batched_attn_mask.unsqueeze(0))
                descriptor_text_features = descriptor_text_features / descriptor_text_features.norm(dim=-1, keepdim=True)
                cls_embeddings = descriptor_text_features
        text_embeddings = torch.stack(text_embeddings, dim=2)
        if self.text_prompt_learner.training:
            sampled_text_features = torch.stack(sampled_text_features, dim=1)
            sampled_frozen_text_features = torch.stack(sampled_frozen_text_features, dim=1)
            sampled_text_features = rearrange(sampled_text_features, 'n_c n d -> (n_c n) d')
            sampled_frozen_text_features = rearrange(sampled_frozen_text_features, 'n_c n d -> (n_c n) d')
            sampled_labels = torch.arange(sampled_frozen_text_features.shape[0], dtype=torch.long).to(device)
        else:
            sampled_text_features = None
            sampled_labels = None
        return cls_embeddings, text_embeddings, sampled_text_features, sampled_frozen_text_features, sampled_labels

class CustomCLIP_ADPE(nn.Module):
    def __init__(self, cfg, clip_model, classnames, lookup_aliases=None):
        super().__init__()
        self.classnames = classnames
        self.text_prompt_learner = TextPromptLearner(cfg, classnames, clip_model, lookup_aliases)
        self.vision_prompt_learner = VisualPromptLearner(cfg, clip_model, self.text_prompt_learner.num_prompts)
        self.text_encoder = TextEncoder(cfg, clip_model, classnames, self.text_prompt_learner, self.vision_prompt_learner.poolings)
        self.visual_encoder = VisualEncoder(cfg, clip_model,  self.text_prompt_learner.num_prompts)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.u1 = cfg.MODEL.ADPE.U1
        self.u3 = cfg.MODEL.ADPE.U3
        self.u2 = cfg.MODEL.ADPE.U2
        self.alpha = cfg.MODEL.ADPE.ALPHA
        self.attr_norm_weight = 1/(self.text_prompt_learner.num_prompts+1)
        self.attr_branch_dropout_p = cfg.MODEL.ADPE.ATTR_BRANCH_DROPOUT_P
        if not 0.0 <= self.attr_branch_dropout_p <= 1.0:
            raise ValueError(
                "ADPE_ATTR_BRANCH_DROPOUT_P must be in [0, 1], "
                f"got {self.attr_branch_dropout_p}."
            )
        self.attr_branch_dropout_rescale = cfg.MODEL.ADPE.ATTR_BRANCH_DROPOUT_RESCALE
        self.attr_dropout_fusion_ce_weight = cfg.MODEL.ADPE.ATTR_DROPOUT_FUSION_CE_WEIGHT
        if self.attr_dropout_fusion_ce_weight < 0.0:
            raise ValueError(
                "ADPE_ATTR_DROPOUT_FUSION_CE_WEIGHT must be non-negative, "
                f"got {self.attr_dropout_fusion_ce_weight}."
            )
        self.last_attr_dropout_mask = None

        if self.attr_branch_dropout_p > 0:
            print(
                f"[ADPE AttrDropout] Enabled attribute branch dropout: "
                f"p={self.attr_branch_dropout_p}, "
                f"rescale={self.attr_branch_dropout_rescale}, "
                f"fusion_ce_weight={self.attr_dropout_fusion_ce_weight}"
            )
        self.use_zs_teacher = self.u1 > 0 or self.u2 > 0
        self.cls_text_features = None
        if self.use_zs_teacher:
            self.zs_clip = load_clip_to_cpu(cfg)
            self.zs_clip.float().cuda()
            cls_temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            cls_prompts = [cls_temp.format(c.replace("_", " ")) for c in classnames]
            cls_prompts = torch.cat([clip.tokenize(p) for p in cls_prompts])
            with torch.no_grad():
                cls_embeddings = self.zs_clip.encode_text(cls_prompts.cuda())
                cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1, keepdim=True)
            self.cls_text_features = cls_embeddings

    def forward(self, image, label=None, alternating_epoch=0, return_dpe_features=False, return_branch_logits=False):
        if self.vision_prompt_learner.training:
            prompt_embeddings, deep_prompt_embeddings = self.vision_prompt_learner()
            cls_img_features, prompt_img_features = self.visual_encoder(image.type(self.dtype), prompt_embeddings, deep_prompt_embeddings)
        else:
            with torch.no_grad():
                prompt_embeddings, deep_prompt_embeddings = self.vision_prompt_learner()
                cls_img_features, prompt_img_features = self.visual_encoder(image.type(self.dtype), prompt_embeddings, deep_prompt_embeddings)
        if len(self.classnames) > 80 and self.text_prompt_learner.training:
            batch_indices = label.detach().cpu().numpy().astype(int).tolist()
            unique_batch_indices = list(OrderedDict.fromkeys(batch_indices))
            unique_batch_set = set(unique_batch_indices)
            remaining_indices = [
                idx for idx in range(len(self.classnames))
                if idx not in unique_batch_set
            ]
            if len(unique_batch_indices) >= 80:
                sampled_indices = unique_batch_indices
            else:
                num_extra = min(
                    80 - len(unique_batch_indices), len(remaining_indices)
                )
                extra_indices = np.random.choice(
                    remaining_indices, num_extra, replace=False
                ).tolist()
                sampled_indices = unique_batch_indices + extra_indices

            sampled_position = {
                old_idx: new_idx
                for new_idx, old_idx in enumerate(sampled_indices)
            }
            new_label = torch.tensor(
                [sampled_position[idx] for idx in batch_indices],
                dtype=torch.long,
                device=cls_img_features.device,
            )
            old_indices = sampled_indices
            zs_text_features = (
                self.cls_text_features[old_indices]
                if self.use_zs_teacher else None
            )
            sampled_classnames = [
                self.text_prompt_learner.lookup_classnames[idx]
                for idx in old_indices
            ]
        else:
            sampled_classnames = None
            new_label = label
            old_indices = None
            zs_text_features = self.cls_text_features
        cls_text_features, prompt_text_features, sampled_text_features, sampled_frozen_text_features, sampled_labels = self.text_encoder(
            prompt_img_features, cls_img_features.unsqueeze(1), alternating_epoch, sampled_classnames, old_indices
        )
        assert prompt_img_features.shape[1] == prompt_text_features.shape[2]
        logit_scale = self.logit_scale.exp()
        cls_img_features = cls_img_features / cls_img_features.norm(dim=-1, keepdim=True)
        cls_text_features = cls_text_features / cls_text_features.norm(dim=-1, keepdim=True)
        cls_logits = logit_scale * torch.bmm(cls_img_features.unsqueeze(1), cls_text_features.transpose(1,2)).squeeze(1)
        prompt_img_features = prompt_img_features / prompt_img_features.norm(dim=-1, keepdim=True)
        prompt_img_features = prompt_img_features.transpose(0,1)
        prompt_text_features = prompt_text_features / prompt_text_features.norm(dim=-1, keepdim=True)
        prompt_text_features = prompt_text_features.permute(2,0,1,3)
        prompt_logits = logit_scale * prompt_img_features.unsqueeze(2)@prompt_text_features.transpose(2,3)
        prompt_logits = prompt_logits.squeeze(2)
        all_logits = torch.cat([cls_logits.unsqueeze(0), prompt_logits], 0)
        if self.use_zs_teacher:
            with torch.no_grad():
                zs_img_features = self.zs_clip.visual(image.type(self.dtype))
                zs_img_features = zs_img_features / zs_img_features.norm(dim=-1, keepdim=True)
            zs_logits = logit_scale * zs_img_features @ zs_text_features.t()
        else:
            zs_img_features = None
            zs_logits = None
        if sampled_text_features is not None:
            assert self.u3 > 0
            sampled_text_features = sampled_text_features / sampled_text_features.norm(dim=-1, keepdim=True)
            sampled_frozen_text_features = sampled_frozen_text_features / sampled_frozen_text_features.norm(dim=-1, keepdim=True)
            tt_logit = logit_scale * sampled_text_features @ sampled_frozen_text_features.t()
            tt_logit_inverse = logit_scale * sampled_frozen_text_features @ sampled_text_features.t()
        else:
            tt_logit = None
            tt_logit_inverse = None
        if self.text_prompt_learner.training or self.vision_prompt_learner.training:
            return all_logits, zs_logits, zs_img_features, cls_img_features, tt_logit, tt_logit_inverse, new_label, sampled_labels
        else:
            if return_branch_logits:
                return all_logits
            final_logits = self.process_logits(all_logits)
            if return_dpe_features:
                attr_img_features = prompt_img_features.permute(1, 0, 2).contiguous()
                return (
                    final_logits,
                    cls_img_features,
                    cls_text_features,
                    attr_img_features,
                )
            return final_logits

    def process_logits(self, all_logits):
        num_attr = all_logits.shape[0] - 1
        device = all_logits.device
        dtype = all_logits.dtype
        if num_attr <= 0:
            raise RuntimeError("ADPE requires at least one attribute branch.")
        attr_weights = torch.ones(num_attr, device=device, dtype=dtype)
        attr_weights = attr_weights / attr_weights.sum().clamp_min(1e-6)
        is_training = self.text_prompt_learner.training or self.vision_prompt_learner.training
        if is_training and self.attr_branch_dropout_p > 0:
            keep_prob = 1.0 - self.attr_branch_dropout_p
            attr_mask = torch.bernoulli(
                torch.full((num_attr,), keep_prob, device=device, dtype=dtype)
            )
            if attr_mask.sum().item() < 1:
                keep_idx = torch.randint(0, num_attr, (1,), device=device)
                attr_mask[keep_idx] = 1.0
            self.last_attr_dropout_mask = attr_mask.detach().float().cpu()
            attr_weights = attr_weights * attr_mask
            if self.attr_branch_dropout_rescale:
                attr_weights = attr_weights / attr_weights.sum().clamp_min(1e-6)
        else:
            self.last_attr_dropout_mask = None
        weights = torch.cat([
            torch.tensor([self.alpha], device=device, dtype=dtype),
            attr_weights * (1.0 - self.alpha)
        ], dim=0).reshape(-1, 1)
        weights = weights.repeat(1, all_logits.shape[1])
        final_logit = torch.einsum('ijk,ij->jk', all_logits, weights.detach())
        return final_logit

@TRAINER_REGISTRY.register()
class ADPE(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADPE.PREC in ("fp32", "amp")
        if not cfg.USE_CUDA or not torch.cuda.is_available():
            raise RuntimeError("The current ADPE implementation requires a CUDA-enabled PyTorch environment")

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.ADPE.PREC == "fp32" or cfg.TRAINER.ADPE.PREC == "amp":

            clip_model.float()
        print("Building custom CLIP for ADPE")
        lookup_aliases = {}
        if _is_imagenet_dataset(cfg.DATASET.NAME):
            synset_aliases = {"n04008634": "projectile", "n04355933": "sunglass",
                              "n04356056": "sunglasses", "n03773504": "missile"}
            for item in self.dm.dataset.test:
                alias = synset_aliases.get(Path(item.impath).parent.name)
                if alias is not None:
                    lookup_aliases[item.label] = alias
        self.model = CustomCLIP_ADPE(cfg, clip_model, classnames, lookup_aliases)
        self.model.to(self.device)
        assert cfg.VISION_OPTIM.MAX_EPOCH + cfg.TEXT_OPTIM.MAX_EPOCH == cfg.OPTIM.MAX_EPOCH
        self.vision_max_epoch = cfg.VISION_OPTIM.MAX_EPOCH
        self.text_max_epoch = cfg.TEXT_OPTIM.MAX_EPOCH
        vision_param_groups = [{'params': self.model.vision_prompt_learner.parameters()}]
        text_param_groups = [{'params': self.model.text_prompt_learner.parameters()}]
        self.vision_optim = build_optimizer(self.model.vision_prompt_learner, cfg.VISION_OPTIM)
        self.text_optim = build_optimizer(self.model.text_prompt_learner, cfg.TEXT_OPTIM)
        self.vision_sched = build_lr_scheduler(self.vision_optim, cfg.VISION_OPTIM)
        self.text_sched = build_lr_scheduler(self.text_optim, cfg.TEXT_OPTIM)
        self.register_model("vision_prompt_learner", self.model.vision_prompt_learner, self.vision_optim, self.vision_sched)
        self.register_model("text_prompt_learner", self.model.text_prompt_learner, self.text_optim, self.text_sched)
        self.scaler = GradScaler() if cfg.TRAINER.ADPE.PREC == "amp" else None
        self.v2t_alternating_epochs = cfg.MODEL.ADPE.V2T_ALTERNATING_EPOCHS
        self.t2v_alternating_epochs = cfg.MODEL.ADPE.T2V_ALTERNATING_EPOCHS
        self.alternating_epoch = 0
        self.vision_epoch = 0
        self.text_epoch = 0
        self.current_phase = cfg.MODEL.ADPE.STARTING_PHASE
        self.setup_phase()
        self.gpa_mean = cfg.MODEL.ADPE.GPA_MEAN
        self.gpa_std = cfg.MODEL.ADPE.GPA_STD
        self.setup_gauss()
        self.previous_vision_gpa = None
        self.previous_text_gpa = None

    def setup_phase(self):
        print(self.current_phase.replace('_', ' '))
        self.turn_off_grad(phase=self.current_phase)
        self._set_phase_model_modes()
        if self.current_phase == "train_text":
            self.model.u1 = 0.0
            self.model.u3 = self.cfg.MODEL.ADPE.U3
        else:
            self.model.u1 = self.cfg.MODEL.ADPE.U1
            self.model.u3 = 0.0

    def _set_phase_model_modes(self):
        self.model.train()
        if self.current_phase == "train_text":
            self._models["vision_prompt_learner"].eval()
            self._models["text_prompt_learner"].train()
        else:
            self._models["vision_prompt_learner"].train()
            self._models["text_prompt_learner"].eval()

    def turn_off_grad(self, phase="train_vision"):
        if phase == "train_vision":
            print("Turning off gradients in both the image and the text encoder, as well as the textual prompts")
            for name, param in self.model.named_parameters():
                if "vision_prompt_learner" not in name and "pooling" not in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
                    param.requires_grad_(True)
        elif phase == "train_text":
            print("Turning off gradients in both the image and the text encoder, as well as the vision prompts")
            for name, param in self.model.named_parameters():
                if "text_prompt_learner" not in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
                    param.requires_grad_(True)
        else:
            raise NotImplementedError(f"phase {phase} not implemented")

    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        test_results = self.after_train()
        return test_results

    def after_train(self):
        print("Finish training")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            test_results = self.test()
        else:
            test_results = None
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        self.close_writer()
        return test_results

    def before_epoch(self):
        if self.current_phase == "train_vision":
            if self.alternating_epoch == self.v2t_alternating_epochs:
                self.alternating_epoch = 0
                self.current_phase = "train_text"
                self.setup_phase()
        else:
            if self.alternating_epoch == self.t2v_alternating_epochs:
                self.alternating_epoch = 0
                self.current_phase = "train_vision"
                self.setup_phase()
    def run_epoch(self):
        self._set_phase_model_modes()
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))
            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()
        self.alternating_epoch += 1
        self.update_gpa()
        if self.current_phase == "train_vision":
            self.vision_epoch += 1
        else:
            self.text_epoch += 1
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        loss_scl_image = None
        loss_tt = None
        loss_tt_inverse = None
        loss_scl_logit = None
        loss_attr_dropout_fusion = None
        with autocast(enabled=self.scaler is not None):
            all_logits, zs_logits, zs_img_features, cls_img_features, tt_logit, tt_logit_inverse, new_label, sampled_labels = self.model(image, label, self.alternating_epoch)
            final_logit = self.model.process_logits(all_logits)
            loss_ce = 0.0
            for logit_idx, logit in enumerate(all_logits):
                loss_ce += self.model.attr_norm_weight * F.cross_entropy(logit, new_label)
            loss = loss_ce
            if self.model.attr_dropout_fusion_ce_weight > 0:
                loss_attr_dropout_fusion = F.cross_entropy(final_logit, new_label)
                loss += self.model.attr_dropout_fusion_ce_weight * loss_attr_dropout_fusion
            if self.model.u1 > 0:
                loss_scl_image = F.l1_loss(cls_img_features, zs_img_features, reduction='mean') * self.model.u1
                loss += loss_scl_image
            if self.model.u3 > 0:
                loss_tt = F.cross_entropy(tt_logit, sampled_labels)
                loss_tt_inverse = F.cross_entropy(tt_logit_inverse, sampled_labels)
                loss_tt_total = self.model.u3 * 0.5 * (loss_tt + loss_tt_inverse)
                loss += loss_tt_total
            if self.model.u2 > 0:
                loss_scl_logit = 0.0
                for logit in all_logits:
                    loss_scl_logit += self.model.attr_norm_weight * F.kl_div(
                        F.log_softmax(logit / 1, dim=1),
                        F.log_softmax(zs_logits / 1, dim=1),
                        reduction='sum',
                        log_target=True
                    ) * (1 * 1) / logit.numel()
                loss += loss_scl_logit * self.model.u2
        self.model_backward_and_update(loss)
        loss_summary = {
            "loss_ce": loss_ce.item()
        }
        if loss_attr_dropout_fusion is not None:
            loss_summary["loss_attr_dropout_fusion"] = loss_attr_dropout_fusion.item()
        if self.model.u1 > 0:
            loss_summary["loss_scl_image"] = loss_scl_image.item()
        if self.model.u3 > 0:
            loss_summary["loss_tt"] = loss_tt.item()
            loss_summary["loss_tt_inverse"] = loss_tt_inverse.item()
        if self.model.u2 > 0:
            loss_summary["loss_scl_logit"] = loss_scl_logit.item()
        loss_summary["acc"] = compute_accuracy(final_logit.float(), new_label)[0].item()
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def get_current_lr(self):
        if self.current_phase == "train_vision":
            name = "vision_prompt_learner"
        else:
            name = "text_prompt_learner"
        return self._optims[name].param_groups[0]["lr"]

    def model_backward_and_update(self, loss):
        if self.current_phase == "train_vision":
            optim = self.vision_optim
        else:
            optim = self.text_optim
        optim.zero_grad()
        self.detect_anomaly(loss)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optim)
            self.scaler.update()
        else:
            loss.backward()
            optim.step()

    def update_lr(self):
        if self.current_phase == "train_vision":
            self.vision_sched.step()
        else:
            self.text_sched.step()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    def _reset_dpe(self):
        self.dpe_cache = {}
        self.dpe_attr_caches = None

    @staticmethod
    def _dpe_cfg(cfg, name, default):
        return getattr(cfg, name, default)

    @staticmethod
    def _dpe_entropy(logits):
        p = logits.softmax(dim=-1)
        return -(p * logits.log_softmax(dim=-1)).sum(dim=-1)

    def _update_dpe_cache(self, features, logits):
        cfg = self.cfg.MODEL.ADPE.DPE
        entropies = self._dpe_entropy(logits).detach()
        predictions = logits.argmax(dim=-1)
        for feature, entropy_value, prediction in zip(features.detach(), entropies, predictions):
            class_id = int(prediction.item())
            item = [feature.unsqueeze(0), float(entropy_value.item())]
            entries = self.dpe_cache.setdefault(class_id, [])
            if len(entries) < cfg.SHOT_CAPACITY:
                entries.append(item)
            elif item[1] < entries[-1][1]:
                entries[-1] = item
            entries.sort(key=operator.itemgetter(1))

    def _ensure_attr_dpe_caches(self, num_attrs):
        if self.dpe_attr_caches is None:
            self.dpe_attr_caches = [{} for _ in range(num_attrs)]
        if len(self.dpe_attr_caches) != num_attrs:
            raise RuntimeError(
                "The number of ADPE attributes changed inside one evaluation stream."
            )

    def _update_attr_dpe_caches(self, attr_features, logits):
        cfg = self.cfg.MODEL.ADPE.DPE
        num_attrs = attr_features.shape[1]
        self._ensure_attr_dpe_caches(num_attrs)
        entropies = self._dpe_entropy(logits).detach()
        predictions = logits.argmax(dim=-1)
        for attr_idx, cache in enumerate(self.dpe_attr_caches):
            branch_features = attr_features[:, attr_idx, :].detach()
            for feature, entropy_value, prediction in zip(
                branch_features, entropies, predictions
            ):
                class_id = int(prediction.item())
                item = [feature.unsqueeze(0), float(entropy_value.item())]
                entries = cache.setdefault(class_id, [])
                if len(entries) < cfg.SHOT_CAPACITY:
                    entries.append(item)
                elif item[1] < entries[-1][1]:
                    entries[-1] = item
                entries.sort(key=operator.itemgetter(1))

    @staticmethod
    def _dpe_make_batch_items(features, entropies, predictions):
        batch_items = []
        for batch_index, (feature, entropy_value, prediction) in enumerate(
            zip(features.detach(), entropies, predictions)
        ):
            batch_items.append(
                (
                    feature.unsqueeze(0),
                    float(entropy_value.item()),
                    int(prediction.item()),
                    batch_index,
                )
            )
        return batch_items

    def _dpe_loo_cache_tensors(
        self,
        persistent_cache,
        batch_items,
        num_classes,
        feature_dim,
        device,
        dtype,
    ):
        cfg = self.cfg.MODEL.ADPE.DPE
        batch_size = len(batch_items)
        keys = torch.zeros(
            batch_size,
            num_classes,
            feature_dim,
            device=device,
            dtype=dtype,
        )
        valid = torch.zeros(
            batch_size, num_classes, device=device, dtype=torch.bool
        )
        batch_by_class = {}
        for item in batch_items:
            batch_by_class.setdefault(item[2], []).append(item)
        active_classes = sorted(set(persistent_cache) | set(batch_by_class))
        for class_id in active_classes:
            candidates = [
                (entry[0], entry[1], None)
                for entry in persistent_cache.get(class_id, [])
            ]
            candidates.extend(
                (feature, entropy_value, batch_index)
                for feature, entropy_value, _, batch_index
                in batch_by_class.get(class_id, [])
            )
            candidates.sort(key=operator.itemgetter(1))
            default_selected = candidates[:cfg.SHOT_CAPACITY]
            if default_selected:
                default_prototype = torch.cat(
                    [entry[0] for entry in default_selected], dim=0
                ).mean(dim=0)
                keys[:, class_id] = F.normalize(default_prototype, dim=-1)
                valid[:, class_id] = True
            for _, _, _, excluded_index in batch_by_class.get(class_id, []):
                selected = [
                    entry for entry in candidates if entry[2] != excluded_index
                ][:cfg.SHOT_CAPACITY]
                if not selected:
                    keys[excluded_index, class_id].zero_()
                    valid[excluded_index, class_id] = False
                    continue
                prototype = torch.cat(
                    [entry[0] for entry in selected], dim=0
                ).mean(dim=0)
                keys[excluded_index, class_id] = F.normalize(
                    prototype, dim=-1
                )
                valid[excluded_index, class_id] = True
        return keys, valid

    @staticmethod
    def _dpe_loo_alignment_loss(evolved_keys, adapted_text, valid_mask):
        text_prototypes = F.normalize(adapted_text.mean(dim=0), dim=-1)
        losses = []
        for sample_index in range(evolved_keys.shape[0]):
            class_ids = valid_mask[sample_index].nonzero(as_tuple=False).flatten()
            if class_ids.numel() == 0:
                continue
            sample_keys = evolved_keys[sample_index, class_ids]
            sample_text = text_prototypes[class_ids]
            align_logits = sample_keys @ sample_text.t() / 0.01
            losses.append(
                F.cross_entropy(
                    align_logits,
                    torch.arange(class_ids.numel(), device=evolved_keys.device),
                )
            )
        if not losses:
            return None
        return torch.stack(losses).mean()
    @staticmethod
    def _dpe_dense_cache_logits(affinity, valid_mask, alpha, beta):
        cache_logits = alpha * torch.exp(-beta * (1.0 - affinity))
        return cache_logits * valid_mask.to(cache_logits.dtype)

    def _mean_loo_attribute_affinity(
        self,
        attr_features,
        entropies,
        predictions,
        class_valid,
        num_classes,
    ):
        num_attrs = attr_features.shape[1]
        if num_attrs <= 0:
            raise RuntimeError("ADPE produced no attribute visual features.")
        self._ensure_attr_dpe_caches(num_attrs)
        affinity_sum = torch.zeros(
            attr_features.shape[0],
            num_classes,
            device=attr_features.device,
            dtype=attr_features.dtype,
        )
        for attr_idx, persistent_cache in enumerate(self.dpe_attr_caches):
            branch_features = attr_features[:, attr_idx, :]
            batch_items = self._dpe_make_batch_items(
                branch_features, entropies, predictions
            )
            attr_keys, attr_valid = self._dpe_loo_cache_tensors(
                persistent_cache,
                batch_items,
                num_classes,
                branch_features.shape[-1],
                branch_features.device,
                branch_features.dtype,
            )
            if not torch.equal(attr_valid, class_valid):
                raise RuntimeError(
                    "LOO attribute and class caches are not sample/class aligned."
                )
            affinity_sum.add_(
                torch.einsum('bd,bcd->bc', branch_features, attr_keys)
            )
        return affinity_sum / float(num_attrs)

    def _dpe_inference(self, input):
        cfg = self.cfg.MODEL.ADPE.DPE
        with torch.no_grad():
            (
                tap_logits,
                image_features,
                text_features,
                attr_features,
            ) = self.model(input, return_dpe_features=True)
        image_features = image_features.detach()
        text_features = text_features.detach()
        attr_features = attr_features.detach()
        if int(cfg.SHOT_CAPACITY) <= 0:
            raise ValueError(
                f"DPE.SHOT_CAPACITY must be positive, got {cfg.SHOT_CAPACITY}."
            )

        batch_size = tap_logits.shape[0]
        num_classes = tap_logits.shape[-1]
        feature_dim = image_features.shape[-1]
        expected_shapes = {
            "tap_logits": (batch_size, num_classes),
            "image_features": (batch_size, feature_dim),
            "text_features": (batch_size, num_classes, feature_dim),
        }
        actual_shapes = {
            "tap_logits": tuple(tap_logits.shape),
            "image_features": tuple(image_features.shape),
            "text_features": tuple(text_features.shape),
        }
        if actual_shapes != expected_shapes:
            raise RuntimeError(
                "Unexpected ADPE tensors for LOO-AA-DPE: "
                f"expected {expected_shapes}, got {actual_shapes}."
            )
        if (
            attr_features.ndim != 3
            or attr_features.shape[0] != batch_size
            or attr_features.shape[1] <= 0
            or attr_features.shape[2] != feature_dim
        ):
            raise RuntimeError(
                "attr_features must have shape [B, A, D] with A > 0; "
                f"got {tuple(attr_features.shape)}."
            )

        rho = float(self._dpe_cfg(cfg, "ATTR_AFFINITY_WEIGHT", 0.1))
        if not 0.0 <= rho <= 1.0:
            raise ValueError(
                f"ATTR_AFFINITY_WEIGHT must be in [0, 1], got {rho}."
            )
        use_attr_affinity = rho > 0.0
        device = image_features.device
        entropies = self._dpe_entropy(tap_logits).detach()
        predictions = tap_logits.argmax(dim=-1)
        class_batch_items = self._dpe_make_batch_items(
            image_features, entropies, predictions
        )
        cache_keys, cache_valid = self._dpe_loo_cache_tensors(
            self.dpe_cache,
            class_batch_items,
            num_classes,
            feature_dim,
            device,
            image_features.dtype,
        )
        if use_attr_affinity:
            attr_affinity = self._mean_loo_attribute_affinity(
                attr_features,
                entropies,
                predictions,
                cache_valid,
                num_classes,
            )
        with torch.enable_grad():
            local_text_residue = nn.Parameter(
                torch.zeros(
                    num_classes,
                    feature_dim,
                    device=device,
                    dtype=image_features.dtype,
                )
            )
            local_cache_residue = nn.Parameter(
                torch.zeros(
                    num_classes,
                    feature_dim,
                    device=device,
                    dtype=image_features.dtype,
                )
            )
            optimizer = torch.optim.AdamW([
                {'params': [local_text_residue], 'lr': cfg.TEXT_LR, 'eps': 1e-3, 'weight_decay': 1e-1},
                {'params': [local_cache_residue], 'lr': cfg.IMAGE_LR, 'eps': 1e-3, 'weight_decay': 1e-1},
            ])
            adapted_text = F.normalize(text_features + local_text_residue.unsqueeze(0), dim=-1)
            text_delta = self.model.logit_scale.exp().detach() * torch.einsum(
                'bd,bcd->bc', image_features, adapted_text - text_features
            )
            evolved_keys = F.normalize(
                cache_keys + local_cache_residue.unsqueeze(0), dim=-1
            )
            affinity = torch.einsum('bd,bcd->bc', image_features, evolved_keys)
            if use_attr_affinity:
                affinity = (1.0 - rho) * affinity + rho * attr_affinity
            cache_logits = self._dpe_dense_cache_logits(
                affinity, cache_valid, cfg.ALPHA, cfg.BETA
            )
            adapted_logits = tap_logits.detach() + text_delta + cache_logits
            loss = self._dpe_entropy(adapted_logits).mean()
            alignment_loss = self._dpe_loo_alignment_loss(
                evolved_keys, adapted_text, cache_valid
            )
            if alignment_loss is not None:
                loss = loss + cfg.ALIGN * alignment_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            adapted_text = F.normalize(text_features + local_text_residue.unsqueeze(0), dim=-1)
            text_delta = self.model.logit_scale.exp().detach() * torch.einsum(
                'bd,bcd->bc', image_features, adapted_text - text_features
            )
            evolved_keys = F.normalize(
                cache_keys + local_cache_residue.unsqueeze(0), dim=-1
            )
            affinity = torch.einsum('bd,bcd->bc', image_features, evolved_keys)
            if use_attr_affinity:
                affinity = (1.0 - rho) * affinity + rho * attr_affinity
            cache_logits = self._dpe_dense_cache_logits(
                affinity, cache_valid, cfg.ALPHA, cfg.BETA
            )
            output = tap_logits + text_delta + cache_logits
            self._update_dpe_cache(image_features, tap_logits)
            if use_attr_affinity:
                self._update_attr_dpe_caches(attr_features, tap_logits)
        return output.detach()

    def test(self, split=None):
        self.set_model_mode("eval")
        self.model.eval()
        self.model.text_encoder.update_text_features()
        self.evaluator.reset()
        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader
        print(f"Evaluate on the *{split}* set")
        if self.cfg.MODEL.ADPE.BRANCH_ABLATION.ENABLED:
            return self._test_branch_ablation(data_loader, split)
        use_dpe = getattr(getattr(self.cfg.MODEL, "ADPE", None), "DPE", None)
        use_dpe = bool(use_dpe and use_dpe.ENABLED)
        if use_dpe:
            self._reset_dpe()
            rho = self._dpe_cfg(
                self.cfg.MODEL.ADPE.DPE, "ATTR_AFFINITY_WEIGHT", 0.1
            )
            print(
                "ADPE Attr-Dropout + LOO-AA-DPE is enabled "
                "(full temporary batch cache; self excluded from class/attr "
                f"caches; online state reset; rho={rho})"
            )
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            if use_dpe:
                output = self._dpe_inference(input)
            else:
                with torch.no_grad():
                    output = self.model_inference(input)
            self.evaluator.process(output, label)
        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        return results

    @torch.no_grad()
    def _test_branch_ablation(self, data_loader, split):
        cfg = self.cfg.MODEL.ADPE.BRANCH_ABLATION
        groups = make_ablation_masks(
            self.model.text_prompt_learner.num_prompts,
            [float(value) for value in cfg.KEEP_RATIOS.split(",")],
            cfg.REPEATS, cfg.EXACT_MAX, cfg.SEED,
        )
        if cfg.MASK_CHUNK <= 0:
            raise ValueError("BRANCH_ABLATION.MASK_CHUNK must be positive")
        masks = [mask for _, _, group in groups for mask in group]
        weights = torch.tensor(masks, device=self.device, dtype=torch.float32)
        weights *= (1.0 - self.model.alpha) / weights.shape[1]
        correct = torch.zeros(len(masks), device=self.device, dtype=torch.long)
        total = 0
        for batch in tqdm(data_loader):
            image, label = self.parse_batch_test(batch)
            logits = self.model(image, return_branch_logits=True).float()
            for start in range(0, len(masks), cfg.MASK_CHUNK):
                fused = self.model.alpha * logits[0].unsqueeze(0) + torch.einsum(
                    "ma,abc->mbc", weights[start:start + cfg.MASK_CHUNK], logits[1:]
                )
                correct[start:start + cfg.MASK_CHUNK] += (fused.argmax(-1) == label.unsqueeze(0)).sum(-1)
            total += label.numel()
        if total == 0:
            raise ValueError("Cannot evaluate an empty dataset")
        accuracies = (100.0 * correct.float() / total).cpu().tolist()
        rows = []
        offset = 0
        for ratio, kept, group in groups:
            scores = accuracies[offset:offset + len(group)]
            offset += len(group)
            row = dict(keep_ratio=ratio, kept_attributes=kept, masks=len(group),
                       accuracy=float(np.mean(scores)), std=float(np.std(scores)))
            rows.append(row)
            print(f"[ADPE ablation] keep={ratio}, attributes={kept}, "
                  f"accuracy={row['accuracy']:.2f} +/- {row['std']:.2f}")
        output = Path(self.cfg.OUTPUT_DIR)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "branch_ablation.json").open("w", encoding="utf-8") as stream:
            json.dump(dict(run=cfg.RUN_NAME, split=split, samples=total, seed=cfg.SEED,
                           results=rows, attribute_masks=masks), stream, indent=2)
        with (output / "branch_ablation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return {"accuracy": rows[0]["accuracy"]}
    
    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss
    
    def setup_gauss(self):
        vision_mean = self.vision_max_epoch * self.gpa_mean
        vision_stdev = self.vision_max_epoch * self.gpa_std
        text_mean = self.text_max_epoch * self.gpa_mean
        text_stdev = self.text_max_epoch * self.gpa_std
        vision_gauss = self.get_gauss(vision_mean, vision_stdev)
        text_gauss = self.get_gauss(text_mean, text_stdev)
        self.vision_gauss = np.array([vision_gauss(a) for a in range(1, self.vision_max_epoch + 1)])
        self.vision_gauss = self.vision_gauss / sum(self.vision_gauss)
        self.text_gauss = np.array([text_gauss(a) for a in range(1, self.text_max_epoch + 1)])
        self.text_gauss = self.text_gauss / sum(self.text_gauss)

    def update_gpa(self):
        if self.current_phase == "train_vision":
            curr_vision_epoch_weight = self.vision_gauss[self.vision_epoch]
            curr_vision_model_weight = copy.deepcopy(self.model.vision_prompt_learner.state_dict())
            weighted_vision_state_dict = self.state_dict_weighting(curr_vision_model_weight, curr_vision_epoch_weight)
            if self.previous_vision_gpa is None:
                self.previous_vision_gpa = weighted_vision_state_dict
            else:
                self.previous_vision_gpa = self.state_dict_add(weighted_vision_state_dict, self.previous_vision_gpa)
            if self.vision_epoch == self.vision_max_epoch-1:
                print("Using GPA model for vision final inference...")
                self.model.vision_prompt_learner.load_state_dict(self.previous_vision_gpa)
        else:
            curr_text_epoch_weight = self.text_gauss[self.text_epoch]
            curr_text_model_weight = copy.deepcopy(self.model.text_prompt_learner.state_dict())
            weighted_text_state_dict = self.state_dict_weighting(curr_text_model_weight, curr_text_epoch_weight)
            if self.previous_text_gpa is None:
                self.previous_text_gpa = weighted_text_state_dict
            else:
                self.previous_text_gpa = self.state_dict_add(weighted_text_state_dict, self.previous_text_gpa)
            if self.text_epoch == self.text_max_epoch-1:
                print("Using GPA model for text final inference...")
                self.model.text_prompt_learner.load_state_dict(self.previous_text_gpa)

    def state_dict_weighting(self, main_dict, weightage):
        updated_dict = copy.deepcopy(main_dict)
        for key in main_dict:
            updated_dict[key] = main_dict[key] * weightage
        return updated_dict

    def state_dict_add(self, dict1, dict2):
        modified_dict = dict2
        for key in dict1:
            modified_dict[key] = (modified_dict[key] + dict1[key])
        return modified_dict
