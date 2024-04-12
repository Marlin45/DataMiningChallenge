# DataMiningChallenge
Our submission to the CSE5243 Sp24 data challenge.

## Layout
The main script to run is main.py it will produce a csv of predictions, predictions.csv

main.py - Creates predictions.csv, our predictions over the test data
/scripts - Scripts we wrote to help produce the predictions including data preprocessing and a neural network
/test,/train - train and test data as provided in the OneDrive
/discovery_notebooks - A collection of jupyter-notebooks that we used to develop and test ideas, they haven't been cleaned up or finalized, most work was done in CombinedData.ipynb
model.keras - A model trained by our neural network in ./scripts/nn.py, is loaded by main or can be retrained very quickly training is <1min


## Running the project
If you can get main.py to run you can produces the predictions.csv which should just involve the python dependencies it and the ./scripts have. Everything should be resolved by "pip install"