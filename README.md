#  Simple QSAR Model with PyTorch

This project demonstrates a basic machine learning model that predicts the biological activity of small molecules using PyTorch. It’s based on converting SMILES strings into molecular fingerprints and training a binary classifier.

##  What is QSAR?
QSAR stands for *Quantitative Structure Activity Relationship*. It uses 
the chemical structure of molecules to predict biological activity. This 
is 
a common task in drug discovery and computational chemistry.

##  Technologies
- Python
- PyTorch
- RDKit
- scikit-learn
- Jupyter Notebook

##  Project Structure
- `data/sample_molecules.csv` — toy dataset of SMILES + activity label
- `qsar_notebook.ipynb` — Jupyter notebook to run everything
- `requirements.txt` — Python dependencies

##  How to Run

1. Clone this repo:
```bash
git clone https://github.com/feelgoodfield/simple-qsar-pytorch.git
cd simple-qsar-pytorch
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Open the notebook:
```bash
jupyter notebook
```

4. Run `qsar_notebook.ipynb` step by step.

##  Output
A simple feedforward neural network is trained to predict activity (active vs inactive) with accuracy displayed on test data.

---

## Notes and Improvements

Right now, the model gets 0.00 test accuracy. This is expected because the 
dataset is very small and the model is basic. Here are some ways to 
improve it:

- Use more training data  
- Adjust the model settings (hyperparameters)  
- Try different ways of turning molecules into numbers (fingerprints)  
- Use better evaluation methods for unbalanced data
