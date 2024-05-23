
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
	$(PYTHON_INTERPRETER) -m gdown 1QmJwHYB0Qbu0Vuoq-DuOlkeQdlJFvXTH -O data/fusion.zip
	@echo "Extracting fusion data..."
	unzip -q data/fusion.zip -d data
	rm data/fusion.zip
	rm -rf data/__MACOSX
	
