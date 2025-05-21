
import sys
import os

module_path='./'

if module_path not in sys.path: 
    sys.path.append(module_path)

# pip install realtabformer
import pandas as pd
from realtabformer import REaLTabFormer

import data
from data import *


# Charger les données (par exemple depuis un CSV)
dt = data.my_data()

# Diviser les données en 80% entraînement et 20% test
X, y = dt.access()

# Réinitialiser les indices pour éviter des conflits
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

df = pd.concat([X.iloc[:,:], y.iloc[:]], axis=1)
#path_data=os.path.join(os.environ['TMPDIR'], 'mimic/data/')

#df = pd.read_csv(path_data+"test_table.csv")

# NOTE: Remove any unique identifiers in the
# data that you don't want to be modeled.

# Non-relational or parent table.
rtf_model = REaLTabFormer(
    model_type="tabular",
    gradient_accumulation_steps=4,
    logging_steps=100)

# Fit the model on the dataset.
# Additional parameters can be
# passed to the `.fit` method.
rtf_model.fit(df)

# Save the model to the current directory.
# A new directory `rtf_model/` will be created.
# In it, a directory with the model's
# experiment id `idXXXX` will also be created
# where the artefacts of the model will be stored.
output_path = os.path.join(os.environ['TMPDIR'], 'data/output')
rtf_model.save(output_path+"/rtf_model/")

# Generate synthetic data with the same
# number of observations as the real dataset.
samples = rtf_model.sample(n_samples=len(df))
# Convert the generated samples to a DataFrame
synthetic_data = pd.DataFrame(samples.iloc[:,:-1], columns=df.columns[:-1])
synthetic_y = pd.DataFrame(samples.iloc[:,-1], columns=[df.columns[-1]])


# Save the synthetic data to a CSV file
synthetic_data.to_csv(output_path+"/synthetic_J44.csv", index=False)
synthetic_y.to_csv(output_path+"/synthetic_y.csv", index=False)

#samples.to_csv(output_path+"/synthetic.csv", index=False)

# Load the saved model. The directory to the
# experiment must be provided.
#rtf_model2 = REaLTabFormer.load_from_dir(
#    path="./rtf_model/idXXXX")
    
