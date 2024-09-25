# Unstable Autoregressive Bandits 
 An undergraduate research project based on the experimental validations of the original publication [Autoregressive Bandits](https://arxiv.org/pdf/2212.06251.pdf)

## Setup

First, clone the repository in your system and then install the dependencies running

```$ pip install -r requirements.txt```

Note that [pywin32==302] goes for Windows. To install dependencies on MacOS, try [python-dateutil==2.8.2] instead. 

## Running

To run selected programs for selected simulations from your directory, follow the following command prompt for the terminal:

```$ python3 exp [YOUR_PROGRAM] [YOUR_JSON_FILE_FOR_SIMULATION]```

Ex.: ```$ python3 exp ARB_exp.py Gamma6=095```

Note that you can several simulation settings altogether:

```$ python3 exp ARB_exp.py Gamma6=095 Gamma6=098 Gamma6=0999```
