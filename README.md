This project explores various generative models for image classifications as well as extensions of their methodology. The focus is on the VAE and DDPM architectures.
## Main Notebook
As a main notebook is required for this project, we have provided one: [Main Notebook](https://github.com/Maplewarrior/42186_final_project/blob/main/main_notebook.ipynb). As many of the experiments do not make sense to run in interactive Python, we plotted data and called make commands from the notebook to show how our work was done.

## Requirements
Download requirements:
```bash
make requirements
```

> [!NOTE]  
> You should setup a venv or conda environment for the project

## Downlaod data
To download the data and process it, run the following command:
```bash
make get-data
```
To get the pokemon fusion data, run:
```bash
make get-fusion-data
```

## Checkpoints and samples
All checkpoints and samples can be found on [Google Drive](https://drive.google.com/drive/folders/1QnYuUFRPSfrjByx6hi0V6Qb1WgcFPwsB?usp=drive_link)

## Jobscripts 
We have provided shell scripts to submit jobs to the [DTU HPC cluster](https://www.hpc.dtu.dk/) in [/jobscripts](https://github.com/Maplewarrior/42186_final_project/tree/main/jobscript).

## Other useful make commands
We have also made [make commands](https://github.com/Maplewarrior/42186_final_project/blob/main/Makefile) for training models, sampling and calculating FID-scores. These should be self-contained and examples are also shown in the [main notebook](https://github.com/Maplewarrior/42186_final_project/blob/main/main_notebook.ipynb).
