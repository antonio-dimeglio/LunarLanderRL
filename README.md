# Policy-Based RL - Lunar Lander Environment 
The goal of the project is to implement a Deep Q-Network approach to the Cartpole Environment implementation provided in the Gymnasium Library.

## Usage
The libraries used for this implementation are:

- Gymnasium (agent environment)
- Pytorch (deep learning backend)
- Numpy (data smoothing for plotting)
- Matplotlib (plotting)
- tqdm (loading bar for training)
- Argparse (argument parsing for experiments)

If all these libraries are already installed then the entry point to run all the experiments provided, Experiment.py can be directly ran. Otherwise, an environment.yml file is provided to install the required libraries.

To create an environment for these libraries the ```environment.yml``` file can be used, by pasting the following two lines into the terminal:

```bash
conda env create -f environment.yml
conda activate rlcartpole
```

In terms of possible experiments to run, different options are possible, with a distinct flag to enable the execution of one or more experiments of choices, which can be printed by using the following line

```bash
python Experiment.py --help
```

Additionally, it is possible to define the number of episode for which an episode is ran by using the ```--num_episodes [value]``` flag, so if you wanted to run all experiments for 500 episodes you'd have to write
```bash
python Experiment.py --num_episodes 500 --run_all
```