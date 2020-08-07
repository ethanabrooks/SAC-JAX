# SAC-JAX
A JAX Implementation of the [Soft Actor Critic](https://arxiv.org/pdf/1801.01290.pdf) Algorithm


## Requirements
Best run in docker: `docker build --tag=<your-tag> .`
For requirement details consult `environment.yml` and `Dockerfile`.

## Example run command
```
python main.py pendulum
```
This will select the config specified in `configs/pendulum.json`. Command line arguments can be used to override arguments in config.

## Results
To do...
