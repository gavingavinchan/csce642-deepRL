# CSCE-642: Deep Reinforcement Learning

## Setup

SWIG is required for installing Box2D. It can be installed on Linux by running 
```bash
sudo apt-get install swig build-essential python-dev python3-dev
```
and on Mac by running
```bash
brew install swig
```
or on windows by following the instructions [here](https://open-box.readthedocs.io/en/latest/installation/install_swig.html).

For setting up the environment, we recommend using conda + pip or virtual env + pip. The Python environment required is 3.9.16 (version)

 Install the packages given by
```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Create the Conda Environment

This project uses Conda to manage dependencies. Create the environment using the appropriate file for your operating system:

**For Linux:**
```bash
conda env create -f environment_linux.yml
```

**For Mac:**
```bash
conda env create -f environment_mac.yml
```

### 2. Activate the Conda Environment

Before running the project, you need to activate the `csce642` Conda environment in your terminal:

```bash
conda activate csce642
```

### 3. Run the Project

You can run the reinforcement learning algorithms using the `run.py` script. For example, to run the "random" solver on the "Blackjack-v1" domain, use the following command:

```bash
python run.py --solver random --domain Blackjack-v1 --episodes 1000
```

## Known Issues

### Plotting Errors (`libGL` error)

When running the project with plotting enabled, you may encounter a `libGL` error. This is related to the graphics driver configuration on your system.

**Workaround:**

You can disable plotting by adding the `--no-plots` flag to the run command. This will allow you to run the simulations and get the results without a graphical display.

```bash
python run.py --solver random --domain Blackjack-v1 --episodes 1000 --no-plots
```

**Potential Fix:**

In some cases, installing `mesa-utils` can resolve this issue. You can install it with the following command:

```bash
sudo apt-get install -y mesa-utils
```
However, this may not work for all system configurations.