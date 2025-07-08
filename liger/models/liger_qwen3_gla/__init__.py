from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from liger.models.liger_qwen3_gla.configuration_liger_qwen3_gla import LigerQwen3GLAConfig
from liger.models.liger_qwen3_gla.modeling_liger_qwen3_gla import LigerQwen3GLAForCausalLM, LigerQwen3GLAModel

AutoConfig.register(LigerQwen3GLAConfig.model_type, LigerQwen3GLAConfig)
AutoModel.register(LigerQwen3GLAConfig, LigerQwen3GLAModel)
AutoModelForCausalLM.register(LigerQwen3GLAConfig, LigerQwen3GLAForCausalLM)


__all__ = ['LigerQwen3GLAConfig', 'LigerQwen3GLAForCausalLM', 'LigerQwen3GLAModel']