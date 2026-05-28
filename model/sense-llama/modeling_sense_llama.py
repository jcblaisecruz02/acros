"""
modeling_sense_llama.py -- Residual sense-induction language model.

This model preserves a frozen Llama LM path and adds a gated Backpack-style
sense path:

    logits = lm_head(base_hidden + effective_gate * sense_mix)

With gate_init=0.0 and min_train_gate=0.0, the initialized model is equivalent
to the base LM while still exposing senses/contextualization for adaptation.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel
from transformers.utils import ModelOutput, logging

try:
    from transformers.cache_utils import Cache
except ImportError:  # pragma: no cover - older transformers.
    Cache = None

try:
    from .configuration_sense_llama import SenseLlamaConfig
except (ImportError, SystemError):
    from configuration_sense_llama import SenseLlamaConfig


logger = logging.get_logger(__name__)


class SenseMLP(nn.Module):
    def __init__(self, in_dim: int, intermediate_dim: int, out_dim: int, act_fn, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(in_dim, intermediate_dim)
        self.c_proj = nn.Linear(intermediate_dim, out_dim)
        self.act = act_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SenseNoMixBlock(nn.Module):
    def __init__(self, embed_dim: int, act_fn, dropout: float, layer_norm_eps: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mlp = SenseMLP(embed_dim, embed_dim * 4, embed_dim, act_fn, dropout)
        self.resid_dropout1 = nn.Dropout(dropout)
        self.resid_dropout2 = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.FloatTensor, residual: torch.FloatTensor) -> torch.FloatTensor:
        residual = self.resid_dropout1(hidden_states) + residual
        hidden_states = self.ln_1(residual)
        mlp_out = self.mlp(hidden_states)
        residual = self.resid_dropout2(mlp_out) + residual
        hidden_states = self.ln_2(residual)
        return hidden_states


class SenseNetwork(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_senses: int,
        sense_intermediate_scale: int,
        act_fn,
        dropout: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.num_senses = num_senses
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.block = SenseNoMixBlock(embed_dim, act_fn, dropout, layer_norm_eps)
        self.ln = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.final_mlp = SenseMLP(
            in_dim=embed_dim,
            intermediate_dim=sense_intermediate_scale * embed_dim,
            out_dim=embed_dim * num_senses,
            act_fn=act_fn,
            dropout=dropout,
        )

    def forward(self, input_embeds: torch.FloatTensor) -> torch.FloatTensor:
        residual = self.dropout(input_embeds)
        hidden_states = self.ln(residual)
        hidden_states = self.block(hidden_states, residual)
        senses = self.final_mlp(hidden_states)
        batch, seq_len, _ = senses.shape
        return senses.reshape(batch, seq_len, self.num_senses, self.embed_dim).transpose(1, 2)


class SenseWeightNetwork(nn.Module):
    def __init__(self, num_senses: int, embed_dim: int):
        super().__init__()
        if embed_dim % num_senses != 0:
            raise ValueError(f"hidden_size ({embed_dim}) must be divisible by num_senses ({num_senses}).")
        self.n_embd = embed_dim
        self.num_senses = num_senses
        self.embed_per_sense = embed_dim // num_senses
        self.c_attn = nn.Linear(embed_dim, 2 * num_senses * self.embed_per_sense)
        self.softmax_scale = None

    def forward(self, encoded: torch.FloatTensor) -> torch.FloatTensor:
        batch, seq_len, _ = encoded.shape
        x = self.c_attn(encoded)
        x = x.reshape(batch, seq_len, 2, self.num_senses, self.embed_per_sense)
        q, k = x.unbind(dim=2)

        scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k) * scale
        causal_mask = torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool).triu_(1)
        scores = scores.float().masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(q.dtype)
        return attn


@dataclass
class SenseLlamaLMHeadModelOutput(ModelOutput):
    logits: torch.FloatTensor = None
    contextualization: torch.FloatTensor = None
    senses: torch.FloatTensor = None
    base_hidden_states: torch.FloatTensor = None
    sense_mix: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    backpack_hidden_states: torch.FloatTensor = None
    gate: torch.FloatTensor = None
    effective_gate: torch.FloatTensor = None
    sense_weight_entropy: Optional[torch.FloatTensor] = None
    per_sense_contribution_norms: Optional[torch.FloatTensor] = None
    loss: Optional[torch.Tensor] = None
    loss_unsmoothed: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple] = None


class SenseLlamaLMHeadModel(LlamaPreTrainedModel, GenerationMixin):
    config_class = SenseLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LlamaDecoderLayer", "SenseNoMixBlock"]
    accepts_loss_kwargs = False
    _tied_weights_keys = {
        "lm_head.weight": "model.embed_tokens.weight",
        "word_embeddings.weight": "model.embed_tokens.weight",
    }

    def __init__(self, config: SenseLlamaConfig, llama_model: Optional[LlamaModel] = None):
        super().__init__(config)
        self.model = llama_model if llama_model is not None else LlamaModel(config)
        self.word_embeddings = self.model.embed_tokens
        self.num_senses = config.num_senses
        self.embed_dim = config.hidden_size
        self.min_train_gate = 0.0

        dropout = getattr(config, "attention_dropout", 0.0)
        act_fn = ACT2FN[config.hidden_act]
        layer_norm_eps = config.rms_norm_eps

        self.sense_network = SenseNetwork(
            embed_dim=self.embed_dim,
            num_senses=config.num_senses,
            sense_intermediate_scale=config.sense_intermediate_scale,
            act_fn=act_fn,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.sense_weight_net = SenseWeightNetwork(config.num_senses, self.embed_dim)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(config.gate_init), dtype=torch.float32))
        self.aux_recon_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        if llama_model is None:
            self.post_init()
        else:
            self.sense_network.apply(self._init_weights)
            self.sense_weight_net.apply(self._init_weights)
            self._init_weights(self.lm_head)
            nn.init.eye_(self.aux_recon_proj.weight)
            self.cast_sense_modules_to_backbone_dtype()
            self.tie_weights()

        self.apply_default_freeze()

    @classmethod
    def from_base_lm(
        cls,
        base_lm,
        num_senses: int = 32,
        sense_intermediate_scale: int = 4,
        gate_init: float = 0.0,
        freeze_backbone: bool = True,
        freeze_lm_head: bool = True,
    ):
        if not hasattr(base_lm, "model"):
            raise ValueError("SenseLlamaLMHeadModel.from_base_lm expects a LlamaForCausalLM-style `.model` backbone.")
        cfg_dict = base_lm.config.to_dict()
        for key in [
            "num_senses",
            "sense_intermediate_scale",
            "gate_init",
            "freeze_backbone",
            "freeze_lm_head",
        ]:
            cfg_dict.pop(key, None)
        cfg = SenseLlamaConfig(
            num_senses=num_senses,
            sense_intermediate_scale=sense_intermediate_scale,
            gate_init=gate_init,
            freeze_backbone=freeze_backbone,
            freeze_lm_head=freeze_lm_head,
            **cfg_dict,
        )
        model = cls(cfg, llama_model=base_lm.model)
        output_embeddings = base_lm.get_output_embeddings()
        if output_embeddings is not None and hasattr(output_embeddings, "weight"):
            model.lm_head.weight = output_embeddings.weight
        model.tie_weights()
        model.apply_default_freeze()
        return model

    def cast_sense_modules_to_backbone_dtype(self) -> None:
        backbone_dtype = next(self.model.parameters()).dtype
        self.sense_network = self.sense_network.to(dtype=backbone_dtype)
        self.sense_weight_net = self.sense_weight_net.to(dtype=backbone_dtype)
        self.lm_head = self.lm_head.to(dtype=backbone_dtype)
        # The optional reconstruction projection is applied to float() tensors
        # in the trainer, so keep it in fp32 unless explicitly cast by a caller.
        self.aux_recon_proj = self.aux_recon_proj.to(dtype=torch.float32)

    def apply_default_freeze(self) -> None:
        if getattr(self.config, "freeze_backbone", True):
            self.model.requires_grad_(False)
        if getattr(self.config, "freeze_lm_head", True):
            self.lm_head.requires_grad_(False)
        self.aux_recon_proj.requires_grad_(False)

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def tie_weights(self, *args, **kwargs):
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_lm_head(self):
        return self.lm_head

    def get_num_senses(self) -> int:
        return self.num_senses

    def get_sense_network(self) -> SenseNetwork:
        return self.sense_network

    def set_min_train_gate(self, value: float) -> None:
        self.min_train_gate = float(value)

    def effective_gate_tensor(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        gate = self.gate.to(device=device, dtype=dtype)
        if self.training and self.min_train_gate != 0.0:
            gate = gate + torch.tensor(float(self.min_train_gate), device=device, dtype=dtype)
        return gate

    def can_generate(self):
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "past_key_values": None,
            "use_cache": False,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "reorder_cache"):
            return past_key_values.reorder_cache(beam_idx)
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def _backbone_forward(self, **kwargs):
        if getattr(self.config, "freeze_backbone", True):
            with torch.no_grad():
                return self.model(**kwargs)
        return self.model(**kwargs)

    @staticmethod
    def _contextualization_entropy(contextualization: torch.Tensor) -> torch.Tensor:
        probs = contextualization.float().clamp_min(1e-9)
        return -(probs * probs.log()).sum(dim=-1).mean()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional["Cache"] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_smoothing: float = 0.0,
        return_dict: Optional[bool] = None,
        output_sense_metrics: bool = False,
        **kwargs,
    ) -> SenseLlamaLMHeadModelOutput:
        kwargs.pop("output_hidden_states", None)
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must pass either input_ids or inputs_embeds.")

        sense_input_embeds = self.word_embeddings(input_ids) if input_ids is not None else inputs_embeds
        senses = self.sense_network(sense_input_embeds)

        llama_out = self._backbone_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )
        base_hidden_states = llama_out.last_hidden_state
        contextualization = self.sense_weight_net(base_hidden_states)
        per_sense_mix = contextualization @ senses
        sense_mix = torch.sum(per_sense_mix, dim=1)

        effective_gate = self.effective_gate_tensor(dtype=base_hidden_states.dtype, device=base_hidden_states.device)
        combined_hidden_states = base_hidden_states + effective_gate * sense_mix
        lm_logits = self.lm_head(combined_hidden_states)

        sense_weight_entropy = None
        per_sense_contribution_norms = None
        if output_sense_metrics:
            with torch.no_grad():
                sense_weight_entropy = self._contextualization_entropy(contextualization)
                per_sense_contribution_norms = per_sense_mix.float().norm(dim=-1).mean(dim=(0, 2))

        loss = None
        loss_unsmoothed = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="mean", label_smoothing=label_smoothing)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            with torch.no_grad():
                ce_raw = CrossEntropyLoss(ignore_index=-100, reduction="mean")
                loss_unsmoothed = ce_raw(
                    shift_logits.detach().view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

        output = SenseLlamaLMHeadModelOutput(
            logits=lm_logits,
            contextualization=contextualization,
            senses=senses,
            base_hidden_states=base_hidden_states,
            sense_mix=sense_mix,
            hidden_states=combined_hidden_states,
            backpack_hidden_states=sense_mix,
            gate=self.gate,
            effective_gate=effective_gate,
            sense_weight_entropy=sense_weight_entropy,
            per_sense_contribution_norms=per_sense_contribution_norms,
            loss=loss,
            loss_unsmoothed=loss_unsmoothed,
            past_key_values=llama_out.past_key_values,
        )
        if return_dict:
            return output
        return (
            output.loss,
            output.logits,
            output.contextualization,
            output.senses,
            output.base_hidden_states,
            output.sense_mix,
            output.hidden_states,
            output.gate,
            output.effective_gate,
            output.past_key_values,
        )
