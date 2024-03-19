from .timellm import TimeLLM
from .dlinear import DLinear


model_lookup = {
	"timellm": TimeLLM,
    "dlinear": DLinear,
}
