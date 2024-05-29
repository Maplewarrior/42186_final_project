# Running experiments on HPC cluster
We are using the HPC cluster at the Techincal University of Denmark to run our experiments. More info can be found [here](https://www.hpc.dtu.dk/).

## Accessing the cluster
To access the cluster you need to have a DTU account and be on the DTU network or VPN. SSH into the cluster using the following command:

```sh
ssh login.gbar.dtu.dk
```


## Jobscripts
To be able to run the jobscripts you need to create an `env_vars.sh` and place it in the [jobscrips folder](https://github.com/AndreasLF/42186_final_project/tree/main/jobscripts). Below is an example of the contents of an `env_vars.sh` file:

```sh
export WORKING_DIR=<working_directory>
export VENV_PATH="$WORKING_DIR/venv/bin/activate"
export EMAIL=<email>
export QUEUE="gpua100"
export NUM_CORES=4
export GPU_MODE="num=1:mode=exclusive_process"
export WALLTIME="12:00"
export MEM_GB=16
```

## Submitting a job

### Jobscript: Train model
To submit a job, run the following command in the terminal:

```sh
./submit_train_job.sh MODEL DATATYPE [P_UNCOND_DDPM/VAE_PRIOR]
```

This will populate the `jobscript_template.sh` with the parameters defined in `env_vars.sh` and submit the job.

Parameters:
- `MODEL`: Is the model to train. Either `VAE` or `DDPM`
- `DATATYPE`: Is the type of data to train on. Either `all`, `fusion`, or `original`
- `[P_UNCOND_DDPM/VAE_PRIOR]` (OPTIONAL): If the `DDPM` model is chosen an optional $p_{uncond}$ can be set. If the `VAE` model is chosen, the prior can be chosen from `std_gauss`/`mog`/`vamp`.

### Jobscript: Sample with model
To draw samples with a trained model, run the following:
```sh
./submit_sample_job.sh MODEL_TYPE MODEL_WEIGHTS BATCH_SIZE [VAE_PRIOR]
```

- `MODEL_TYPE`: Is the model to sample from. Either `VAE` or `DDPM`
- `MODEL_WEIGHTS`: Are the full path to the weights file of the trained model.
- `BATCH_SIZE`: Is the number of samples you want to make at once (one batch is saved as one .pt file)
- `[VAE_PRIOR]` (OPTIONAL): VAE-model is chosen you need to specify the prior  `std_gauss`/`mog`/`vamp`.