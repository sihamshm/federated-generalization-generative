import sys

module='model'


if module not in sys.path: 
    sys.path.append(module)

import my_test
from my_test import*

if __name__ == "__main__":
    data_icu = "ICU"
    cv= 0 # int(5) int(10) cross-validation 5 or 10 fold validation
    concat = 'Conactenate'
    model = 'Xgboost'  #'Logistic Regression','Random Forest','Gradient Bossting','Xgboost'
    ml_model= my_test.ML_models(data_icu,cv,model,concat)
