.PHONY: generate_solutions evaluate_solutions

generate_solutions:
	python generate_solutions.py

evaluate_solutions:
	python -m human_eval.evaluate_functional_correctness samples.jsonl

set-env:
	deactivate
	pyenv activate evals