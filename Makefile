
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

train-vae:
	$(PYTHON_INTERPRETER) main.py train --model-type VAE --data-type $(DATA_TYPE) --vae-prior $(VAE_PRIOR)
train-ddpm:
	$(PYTHON_INTERPRETER) main.py train --model-type DDPM --data-type $(DATA_TYPE) --p-uncond $(P_UNCOND)

get-data:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/data_utils/get_data.py

scrape-fusion-data:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/data_utils/scrape_fusion.py

test-dataloader:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/data_utils/dataloader.py

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