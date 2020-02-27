# Stylized Model of Smallholder Agricultural Households
Author details will be revealed upon acceptance of the publication.

A COMPLETE ODD PROTOCOL WILL BE RELEASED UPON ACCEPTANCE OF THE PUBLICATION.

## Model overview
To be added upon acceptance of the publication.

## 1. Code structure
The code is divided into a few folders:

`model/` contains all the model code. The model can be run as described below. All model inputs are in the `base_inputs.py` file.

`experiments/` contains code for running various experiments (e.g., shock simulations, scenario analysis, sensitivity analysis).

`calibration/` contains the code for pattern-oriented modeling.

`plot/` contains all plotting functions.

NOTE: while most files are `.py`, some are `.ipynb`, which are Jupyter Notebook files.

## 2. Run the model
Requirements:
- Python >=3.6
- Make sure all required packages in the `requirements.txt` file are installed.
- Create a folder called `code` for the code and a folder called `outputs/` for the outputs to be stored.

Navigate to the `code` directory.

From the command line, type:

`python run.py`

This will run the model and some plotting functions.

## 3. Conduct experiments
The experiments conducted for the main body of the paper are coded in the `experiments/analysis.py` and `experiments/multi_scenario_analysis.py` files.
