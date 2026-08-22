#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = diffusion_inversion_for_image_editing
PYTHON_VERSION = 3.11
UV = uv
UV_CACHE_DIR ?= $(CURDIR)/cache/uv
PYTHON_INTERPRETER = $(UV) run python
TRAJ_OUTPUT_DIR ?= data/processed/sdxl_trajectories
TRAJ_NUM_SAMPLES ?= 4
TRAJ_START_INDEX ?= 0
TRAJ_SEED ?= 1234
TRAJ_GUIDANCE_SCALE ?= 1.0
TRAJ_NUM_INFERENCE_STEPS ?= 50
INVERT_OVERRIDES ?=
RECON_OVERRIDES ?=
EVAL_OVERRIDES ?=

export UV_CACHE_DIR

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	$(UV) run ruff format --check
	$(UV) run ruff check

## Format source code with ruff
.PHONY: format
format:
	$(UV) run ruff check --fix
	$(UV) run ruff format


## Download Recap-COCO from Hugging Face using config/data/recap_coco.yaml
.PHONY: data-download-recap-coco
data-download-recap-coco:
	$(UV) run python -m diff_inversion.data.download_recap_coco

## Prepare processed Recap-COCO prompt splits
.PHONY: data-prepare-recap-coco
data-prepare-recap-coco:
	$(UV) run python -m diff_inversion.data.prepare_recap_coco_prompts

## Run the full Recap-COCO preparation pipeline
.PHONY: data-recap-coco
data-recap-coco: data-download-recap-coco data-prepare-recap-coco

## Generate baseline SDXL forward trajectories and predicted noises
.PHONY: generate_trajectories generate-baseline-samples
generate_trajectories:
	$(UV) run python -m diff_inversion.data.generate_sdxl_samples \
		output_dir=$(TRAJ_OUTPUT_DIR) \
		num_samples=$(TRAJ_NUM_SAMPLES) \
		start_index=$(TRAJ_START_INDEX) \
		seed=$(TRAJ_SEED) \
		model.guidance_scale=$(TRAJ_GUIDANCE_SCALE) \
		model.num_inference_steps=$(TRAJ_NUM_INFERENCE_STEPS)

generate-baseline-samples: generate_trajectories


## Run optional baseline DDIM inversion over generated SDXL trajectory samples
.PHONY: invert-baseline-samples
invert-baseline-samples:
	$(UV) run python -m diff_inversion.eval.invert_sdxl $(INVERT_OVERRIDES)


## Reconstruct baseline SDXL images from saved inverted DDIM noise
.PHONY: reconstruct-baseline-samples
reconstruct-baseline-samples:
	$(UV) run python -m diff_inversion.eval.reconstruct_sdxl $(RECON_OVERRIDES)


## Run lightweight evaluation over generated SDXL trajectory artifacts
.PHONY: evaluate evaluate-wandb
evaluate:
	$(UV) run python -m diff_inversion.eval.run $(EVAL_OVERRIDES)

## Run evaluation and log metrics to Weights & Biases
evaluate-wandb:
	$(UV) run python -m diff_inversion.eval.run wandb.mode=online $(EVAL_OVERRIDES)





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
