from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from liger.models.liger_qwen2_gla.configuration_liger_qwen2_gla import LigerQwen2GLAConfig
from liger.models.liger_qwen2_gla.modeling_liger_qwen2_gla import LigerQwen2GLAForCausalLM, LigerQwen2GLAModel

AutoConfig.register(LigerQwen2GLAConfig.model_type, LigerQwen2GLAConfig)
AutoModel.register(LigerQwen2GLAConfig, LigerQwen2GLAModel)
AutoModelForCausalLM.register(LigerQwen2GLAConfig, LigerQwen2GLAForCausalLM)


__all__ = ['LigerQwen2GLAConfig', 'LigerQwen2GLAForCausalLM', 'LigerQwen2GLAModel']