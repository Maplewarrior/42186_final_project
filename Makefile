
.PHONY: requirements dev_requirements clean data build_documentation serve_documentation

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = src
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_venv:
	$(PYTHON_INTERPRETER) -m venv venv

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

DATA_TYPE = all
VAE_PRIOR = std_gauss
P_UNCOND = 0.1
MODEL_WEIGHTS = weights/DDPM_weights_344b68a7-1516-411d-898b-fb3a643dfb02.pt
# MODEL_WEIGHTS = checkpoints/DDPM/344b68a7-1516-411d-898b-fb3a643dfb02/checkpoint_599epochs.pt
MODEL_WEIGHTS = checkpoints/VAE/f05c86aa-7700-4020-bd6d-51f0a99dc598/checkpoint_399epochs.pt

train-vae:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/main.py train --model-type VAE --data-type $(DATA_TYPE) --vae-prior $(VAE_PRIOR)

train-ddpm:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/main.py train --model-type DDPM --data-type $(DATA_TYPE) --p-uncond $(P_UNCOND)

sample-vae:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/main.py sample --model-type VAE --data-type $(DATA_TYPE) --vae-prior $(VAE_PRIOR) \
	 --load-weights $(MODEL_WEIGHTS) --num-samples 1000

sample-ddpm:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/main.py sample --model-type DDPM --data-type $(DATA_TYPE) --p-uncond $(P_UNCOND) \
	--load-weights $(MODEL_WEIGHTS) --num-samples 1000

get-data:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/data_utils/get_data.py

scrape-fusion-data:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/data_utils/scrape_fusion.py

test-dataloader:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/data_utils/dataloader.py

test-samples-dataloader:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/data_utils/samples_dataloader.py

test-metadata:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/data_utils/metadata.py

get-fusion-data:
	@echo "Downloading fusion data..."
	$(PYTHON_INTERPRETER) -m gdown 1QmwBJNZjsPcnBnBBbjOw6zRT95zeviB3 -O data/fusion.zip
	@echo "Extracting fusion data..."
	unzip -q data/fusion.zip -d data
	rm data/fusion.zip
	rm -rf data/__MACOSX
	
# This is just an example implementation so far
calculate-fid:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/calculate_fid.py