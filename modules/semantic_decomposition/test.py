import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.semantic_decomposition.semantic_decomposer import SemanticDecomposer

decomposer = SemanticDecomposer()

cases = [
	"Who is the CEO of Google and solve 2x + 4 = 10",
	"Name a famous person who embodies the following values: knowledge and creativity.",
	"Find X, then use that result to calculate Y.",
	"Compare and contrast two popular tourist attractions in your hometown.",
	"Write a haiku about rain and translate 'hello world' to French.",
]


for prompt in cases:
	tasks = decomposer.decompose(prompt)
	print("PROMPT:", prompt)
	print("TASKS:", tasks)
	print()

assert len(decomposer.decompose(cases[0])) == 2
assert len(decomposer.decompose(cases[1])) == 1
assert decomposer.decompose(cases[2])[1]["depends_on_previous"] is True
assert len(decomposer.decompose(cases[3])) == 1