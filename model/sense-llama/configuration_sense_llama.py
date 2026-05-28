from transformers.models.llama.configuration_llama import LlamaConfig


class SenseLlamaConfig(LlamaConfig):
    """
    Configuration for a residual sense-induction Llama language model.

    Inherits all Llama parameters and adds:
        num_senses: number of sense vectors per token.
        sense_intermediate_scale: MLP width multiplier in the sense network.
        gate_init: initial raw scalar residual gate.
        freeze_backbone: whether the Llama backbone is frozen by default.
        freeze_lm_head: whether the LM head is frozen by default.
    """

    model_type = "sense_llama"

    def __init__(
        self,
        num_senses: int = 32,
        sense_intermediate_scale: int = 4,
        gate_init: float = 0.0,
        freeze_backbone: bool = True,
        freeze_lm_head: bool = True,
        **kwargs,
    ):
        self.num_senses = num_senses
        self.sense_intermediate_scale = sense_intermediate_scale
        self.gate_init = gate_init
        self.freeze_backbone = freeze_backbone
        self.freeze_lm_head = freeze_lm_head
        super().__init__(**kwargs)
        self.auto_map = {
            "AutoConfig": "configuration_sense_llama.SenseLlamaConfig",
            "AutoModelForCausalLM": "modeling_sense_llama.SenseLlamaLMHeadModel",
        }
