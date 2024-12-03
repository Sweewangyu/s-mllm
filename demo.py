from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    Blip2ForConditionalGeneration,
)

# Initializing a Blip2Config with Salesforce/blip2-opt-2.7b style configuration
# configuration = Blip2Config()
#
# # Initializing a Blip2ForConditionalGeneration (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
# model = Blip2ForConditionalGeneration(configuration)
#
# # Accessing the model configuration
# configuration = model.config

# We can also initialize a Blip2Config from a Blip2VisionConfig, Blip2QFormerConfig and any PretrainedConfig

# Initializing BLIP-2 vision, BLIP-2 Q-Former and language model configurations
vision_config = Blip2VisionConfig()
qformer_config = Blip2QFormerConfig()
text_config = OPTConfig()

config = Blip2Config.from_vision_qformer_text_configs(vision_config, qformer_config, text_config)
model = Blip2ForConditionalGeneration(config)
print(model)