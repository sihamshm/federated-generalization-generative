import sys

module_path='model'

if module_path not in sys.path: 
    sys.path.append(module_path)
    
import ml_models
from ml_models import*




if __name__ == "__main__":
    data_icu = "ICU"
    cv= int(3)# int(5) int(10) cross-validation 5 or 10 fold validation
    concat = 'Conactenate'
    oversampling = 'True'
    model = 'Logistic Regression'  #'Logistic Regression','Random Forest','Gradient Bossting','Xgboost'
    ml=ml_models.ML_models(data_icu,cv,model,concat,oversampling)

    
