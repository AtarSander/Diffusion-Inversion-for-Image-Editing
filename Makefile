#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = diffusion_inversion_for_image_editing
PYTHON_VERSION = 3.11
UV = uv
UV_CACHE_DIR = $(CURDIR)/.uv-cache
PYTHON_INTERPRETER = $(UV) run python

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

.PHONY: generate_trajectories
generate_trajectories:
	$(UV) run python -m diff_inversion.data.generate_sdxl_samples





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
