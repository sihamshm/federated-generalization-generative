import pandas as pd
import numpy as np
import pickle
import torch
import random
import os
import importlib
import sys
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')


import evaluation

import joblib

# MAX_LEN=12
# MAX_COND_SEQ=56
# MAX_PROC_SEQ=40
# MAX_MED_SEQ=15#37
# MAX_LAB_SEQ=899
# MAX_BMI_SEQ=118


class ML_models():
    def __init__(self, data_icu, k_fold, model_type, concat):
        self.data_icu = data_icu
        self.k_fold = 1  # Pas de k-fold, tout sera considéré comme test
        self.model_type = model_type
        self.concat = concat
        self.loss = evaluation.Loss('cpu', True, True, True, True, True, True, True, True, True, True, True)
        self.ml_train()
    def test_model(self,X_test,Y_test):
        # Charger un modèle avec joblib
        model = joblib.load('./data/output/trained_model.pkl')
        print("les caractéristiques de mon modèle",model.get_params())

        print("===============MODEL TRAINING===============")
        if self.model_type=='Gradient Bossting':
            prob=model.predict_proba(X_test)
            logits=np.log2(prob[:,1]/prob[:,0])
            self.loss(prob[:,1],np.asarray(Y_test),logits,False,True)
            self.save_output(Y_test,prob[:,1],logits)
        
        elif self.model_type=='Logistic Regression':
            logits=model.predict_log_proba(X_test)
            prob=model.predict_proba(X_test)
            self.loss(prob[:,1],np.asarray(Y_test),logits[:,1],False,True)
            self.save_outputImp(Y_test,prob[:,1],logits[:,1],model.coef_[0],X_test.columns)
        
        elif self.model_type=='Random Forest':
            logits=model.predict_log_proba(X_test)
            prob=model.predict_proba(X_test)
            self.loss(prob[:,1],np.asarray(Y_test),logits[:,1],False,True)
            self.save_outputImp(Y_test,prob[:,1],logits[:,1],model.feature_importances_,X_test.columns)
    
        elif self.model_type=='Xgboost':
            prob=model.predict_proba(X_test)
            logits=np.log2(prob[:,1]/prob[:,0])
            self.loss(prob[:,1],np.asarray(Y_test),logits,False,True)
            self.save_outputImp(Y_test,prob[:,1],logits,model.feature_importances_,X_test.columns)


    def getXY(self,ids,labels,concat_cols):
        X_df=pd.DataFrame()   
        y_df=pd.DataFrame()   
        features=[]
        #print(ids)
        for sample in ids:
            if self.data_icu:
                y=labels[labels['stay_id']==sample]['label']
            else:
                y=labels[labels['hadm_id']==sample]['label']
            
            #print(sample)
            dyn=pd.read_csv('./data/csv/'+str(sample)+'/dynamic.csv',header=[0,1])
            
            if self.concat:
                dyn.columns=dyn.columns.droplevel(0)
                dyn=dyn.to_numpy()
                dyn=dyn.reshape(1,-1)
                #print(dyn.shape)
                #print(len(concat_cols))
                dyn_df=pd.DataFrame(data=dyn,columns=concat_cols)
                features=concat_cols
            else:
                dyn_df=pd.DataFrame()
                #print(dyn)
                for key in dyn.columns.levels[0]:
                    #print(sample)                    
                    dyn_temp=dyn[key]
                    if self.data_icu:
                        if ((key=="CHART") or (key=="MEDS")):
                            agg=dyn_temp.aggregate("mean")
                            agg=agg.reset_index()
                        else:
                            agg=dyn_temp.aggregate("max")
                            agg=agg.reset_index()
                    else:
                        if ((key=="LAB") or (key=="MEDS")):
                            agg=dyn_temp.aggregate("mean")
                            agg=agg.reset_index()
                        else:
                            agg=dyn_temp.aggregate("max")
                            agg=agg.reset_index()
                    if dyn_df.empty:
                        dyn_df=agg
                    else:
                        dyn_df=pd.concat([dyn_df,agg],axis=0)
                #dyn_df=dyn_df.drop(index=(0))
#                 print(dyn_df.shape)
#                 print(dyn_df.head())
                dyn_df=dyn_df.T
                dyn_df.columns = dyn_df.iloc[0]
                dyn_df=dyn_df.iloc[1:,:]
                        
#             print(dyn.shape)
#             print(dyn_df.shape)
#             print(dyn_df.head())
            stat=pd.read_csv('./data/csv/'+str(sample)+'/static.csv',header=[0,1])
            stat=stat['COND']
#             print(stat.shape)
#             print(stat.head())
            demo=pd.read_csv('./data/csv/'+str(sample)+'/demo.csv',header=0)
#             print(demo.shape)
#             print(demo.head())
            if X_df.empty:
                X_df=pd.concat([dyn_df,stat],axis=1)
                X_df=pd.concat([X_df,demo],axis=1)
            else:
                X_df=pd.concat([X_df,pd.concat([pd.concat([dyn_df,stat],axis=1),demo],axis=1)],axis=0)
            if y_df.empty:
                y_df=y
            else:
                y_df=pd.concat([y_df,y],axis=0)
#             print("X_df",X_df.shape)
#             print("y_df",y_df.shape)
        print("X_df",X_df.shape)
        print("y_df",y_df.shape)
        return X_df ,y_df
    
    def save_output(self,labels,prob,logits):
        
        output_df=pd.DataFrame()
        output_df['Labels']=labels.values
        output_df['Prob']=prob
        output_df['Logits']=np.asarray(logits)
        output_df['ethnicity']=list(self.test_data['ethnicity'])
        output_df['gender']=list(self.test_data['gender'])
        output_df['age']=list(self.test_data['Age'])
        output_df['insurance']=list(self.test_data['insurance'])
        
        with open('./data/output/'+'outputDict', 'wb') as fp:
               pickle.dump(output_df, fp)
        
    
    def save_outputImp(self,labels,prob,logits,importance,features):
        
        output_df=pd.DataFrame()
        output_df['Labels']=labels.values
        output_df['Prob']=prob
        output_df['Logits']=np.asarray(logits)
        output_df['ethnicity']=list(self.test_data['ethnicity'])
        output_df['gender']=list(self.test_data['gender'])
        output_df['age']=list(self.test_data['Age'])
        output_df['insurance']=list(self.test_data['insurance'])
        
        with open('./data/output/'+'outputDict', 'wb') as fp:
               pickle.dump(output_df, fp)
        
        imp_df=pd.DataFrame()
        imp_df['imp']=importance
        imp_df['feature']=features
        imp_df.to_csv('./data/output/'+'feature_importance.csv', index=False)
                
    def ml_train(self):
        labels = pd.read_csv('./data/csv/labels.csv', header=0)

        # Utilisation de toutes les données comme données de test
        test_hids = labels.iloc[:, 0]
        print("=================== TEST DATA =====================")
        
        concat_cols = []
        if self.concat:
            dyn = pd.read_csv('./data/csv/' + str(test_hids[0]) + '/dynamic.csv', header=[0, 1])
            dyn.columns = dyn.columns.droplevel(0)
            cols = dyn.columns
            time = dyn.shape[0]

            for t in range(time):
                cols_t = [x + "_" + str(t) for x in cols]
                concat_cols.extend(cols_t)

        print('test_hids', len(test_hids))

        # Les données de test
        X_test, Y_test = self.getXY(test_hids, labels, concat_cols)
        
        #récupérer les données X_test1 de demo1
        X_test1 = pd.DataFrame()
        demo_path =  "./data/csv/"
        for sample in test_hids:
            demo=pd.read_csv(demo_path+str(sample)+'/demo1.csv',header=0)
            if X_test1.empty:
                X_test1=pd.concat([X_test1,demo],axis=1)
            else:
                X_test1=pd.concat([X_test1,demo],axis=0)
        
        self.test_data=X_test1.copy(deep=True)
        # Test avec le modèle
        self.test_model(X_test, Y_test)

    




