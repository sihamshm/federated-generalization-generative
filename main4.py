import sys

module_path='model'

if module_path not in sys.path: 
    sys.path.append(module_path)
    
import ml_models4
from ml_models4 import*




if __name__ == "__main__":
    data_icu = "ICU"
    cv= int(3) # int(5) int(10) cross-validation 5 or 10 fold validation
    concat = 'Conactenate'
    oversampling = 'True'
    model = 'Xgboost'  #'Logistic Regression','Random Forest','Gradient Bossting','Xgboost'
    ml=ml_models4.ML_models(data_icu,cv,model,concat,oversampling)

    
