from .contract_template import ContractsPromptsTemplate
from .math_template import MathPromptsTemplate

prompt_template = "math_template"

PROMPT_TEMPLATE_MAP = {
    "math_template": MathPromptsTemplate, 
    "contract_template": ContractsPromptsTemplate,
}
