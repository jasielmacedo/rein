NEURON_ALIGN = 256

CONFIG_FILE = "config.json"
INDEX_FILE = "model.safetensors.index.json"
SAFETENSORS_SUFFIX = ".safetensors"
MANIFEST_FILE = "rein_neurons.json"
MANIFEST_VERSION = 1
SAFETENSORS_METADATA = {"format": "pt"}

TEXT_CONFIG = "text_config"
INTERMEDIATE_SIZE = "intermediate_size"
HIDDEN_SIZE = "hidden_size"
NUM_HIDDEN_LAYERS = "num_hidden_layers"

MLP_KEY_PATTERN = (
    r"^(?P<scope>model\.language_model\.layers|model\.layers|mtp\.layers)\.(?P<layer>\d+)"
    r"\.mlp\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
MTP_SCOPE = "mtp.layers"
GATE_PROJ = "gate_proj"
UP_PROJ = "up_proj"
DOWN_PROJ = "down_proj"

SIGNAL_TOKENS = (
    "<|repo_name|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|file_sep|>",
)

DEFAULT_CHANNEL_RIDGE = 1.0e-3
