from .timellm import TimeLLM
from .gpt4ts import GPT4TS

from .dlinear import DLinear
from .FEDformer import FEDformer


model_lookup = {
	"timellm": TimeLLM,
	"gpt4ts": GPT4TS,
    "dlinear": DLinear,
    "fedformer": FEDformer,
}
