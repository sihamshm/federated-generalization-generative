# ============================================================================
# Ce script est une version modifiée d’un des composants du pipeline initial publié par :
# M. Gupta, B. M. Gallamoza, N. Cutrona, P. Dhakal, R. Poulain, and R. Beheshti,
# “An extensive data processing pipeline for MIMIC-IV,” in Proceedings of the
# 2nd Machine Learning for Health symposium, PMLR 193:311–325, 2022.
# Dépôt original : https://github.com/healthylaife/MIMIC-IV-Data-Pipeline
#
# Licence d’origine : MIT License
#
# Modifications apportées par : [sihamshm]
# Contexte : Projet de recherche sur l’évaluation du potentiel des méthodes génératives pour améliorer la généralisation en apprentissage fédéré
# Description des modifications : 
# - Séparation de l’étape de prétraitement des données de celle de l’entraînement des modèles
# -Réduction de la redondance dans le traitement des données en modifiant le pipeline original qui récupérait les données à chaque
# itération. Désormais, les données sont extraites et enregistrées une seule fois dans un DataFrame avant les boucles de traitement.
# ============================================================================
import pandas as pd
import numpy as np
import pickle
import torch
import random
import os
import importlib
import sys
import numpy as np
import evaluation4
import joblib
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

#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
sys.path.append(os.path.join(os.environ['TMPDIR'], 'mimic'))
sys.path.append(os.path.join(os.environ['TMPDIR'], 'data4/output'))


importlib.reload(evaluation4)
import evaluation4
# MAX_LEN=12
# MAX_COND_SEQ=56
# MAX_PROC_SEQ=40
# MAX_MED_SEQ=15#37
# MAX_LAB_SEQ=899
# MAX_BMI_SEQ=118


class ML_models():
    def __init__(self,data_icu,k_fold,model_type,concat,oversampling):
        self.data_icu=data_icu
        self.k_fold=k_fold
        self.model_type=model_type
        self.concat=concat
        self.oversampling=oversampling
        self.fold=0
        self.loss=evaluation4.Loss('cpu',True,True,True,True,True,True,True,True,True,True,True)
        self.ml_train()
    def create_kfolds(self):
        labels_path = os.path.join(os.environ['TMPDIR'], 'mimic/data/csv/labels.csv')
        labels=pd.read_csv(labels_path, header=0)
        
        if (self.k_fold==0):
            k_fold=5
            self.k_fold=1
        else:
            k_fold=self.k_fold
        hids=labels.iloc[:,0]
        y=labels.iloc[:,1]
        #print("Total Samples",len(hids))
        #print("Positive Samples",y.sum())
        #print(len(hids))
        output_path = os.path.join(os.environ['TMPDIR'], 'data4/output')
        with open(os.path.join(output_path,"model_results.txt"), "a") as file:
            file.write("Total Samples: {:.2f} \n".format(len(hids)))
            file.write("Positive Samples: {:.2f} \n".format(y.sum()))
        """
        if self.oversampling:
            print("=============OVERSAMPLING===============")
            oversample = RandomOverSampler(sampling_strategy='minority')
            hids=np.asarray(hids).reshape(-1,1)
            hids, y = oversample.fit_resample(hids, y)
            #print(hids.shape)
            hids=hids[:,0]
            print("Total Samples",len(hids))
            print("Positive Samples",y.sum())
        """
        ids=range(0,len(hids))
        batch_size=int(len(ids)/k_fold)
        k_hids=[]
        for i in range(0,k_fold):
            rids = random.sample(ids, batch_size)
            ids = list(set(ids)-set(rids))
            if i==0:
                k_hids.append(hids[rids])             
            else:
                k_hids.append(hids[rids])
        return k_hids


    def ml_train(self):
        k_hids=self.create_kfolds()
        labels_path = os.path.join(os.environ['TMPDIR'], 'mimic/data/csv/labels.csv')       
        labels=pd.read_csv(labels_path, header=0)
        concat_cols=[]
        if(self.concat):
            dyn_path = os.path.join(os.environ['TMPDIR'], 'mimic/data/csv/')
            dyn = pd.read_csv(dyn_path+str(k_hids[0].iloc[0])+'/dynamic.csv',header=[0,1])
            dyn.columns=dyn.columns.droplevel(0)
            cols = dyn.columns
            time = dyn.shape[0]

            for t in range(time):
                cols_t = [x + "_"+str(t) for x in cols]
                concat_cols.extend(cols_t)
        #print('train_hids',len(train_hids))

        total_hids = []

        for j in range(len(k_hids)):
            total_hids.extend(k_hids[j]) 
        X,Y = self.getXY(total_hids,labels,concat_cols)
     


        #récupérer les données X_test1 de demo1
        
        for i in range(self.k_fold):
            print("==================={0:2d} FOLD=====================".format(i))
            test_hids=k_hids[i].tolist()
            train_ids=list(set([0,1,2])-set([i]))
            train_hids=[]
            for j in train_ids:
                train_hids.extend(k_hids[j])                    

            indices_test_hids = []
            for id in test_hids:
                # Trouver le premier indice de id dans total_hids (s'il est présent)
                index = total_hids.index(id)
                indices_test_hids.append(index)
            #print("indices_test_hids", indices_test_hids)



            X_test = X.iloc[indices_test_hids]

            # Crée une copie de labels où stay_id est dans test_hids pour éviter l'avertissement
            labels_test = labels[labels['stay_id'].isin(test_hids)].copy()
            
            # Transformer 'stay_id' en une catégorie ordonnée selon test_hids
            labels_test['stay_id'] = pd.Categorical(labels_test['stay_id'], categories=test_hids, ordered=True)
            
            
            # Trier labels_test pour respecter l'ordre de test_hids
            labels_test = labels_test.sort_values('stay_id').reset_index(drop=True)

            y_test = labels_test.iloc[:,1]
            y_test.fillna(0, inplace=True)  # Remplacer les NaN par 0 dans Y_train



            X_test1 = pd.DataFrame()
            demo_path =  os.path.join(os.environ['TMPDIR'], 'mimic/data/csv/')
            for sample in test_hids:
                demo = pd.read_csv(demo_path+str(sample)+'/demo1.csv',header=0)
                if X_test1.empty:
                    X_test1 = pd.concat([X_test1,demo],axis=1)
                else:
                    X_test1 = pd.concat([X_test1,demo],axis=0)
            
            self.test_data = X_test1.copy(deep=True)




            # Crée une copie de labels où stay_id est dans test_hids pour éviter l'avertissement
            labels_train = labels[labels['stay_id'].isin(train_hids)].copy()
            
            # Transformer 'stay_id' en une catégorie ordonnée selon test_hids
            labels_train['stay_id'] = pd.Categorical(labels_train['stay_id'], categories=train_hids, ordered=True)
            
            
            # Trier labels_test pour respecter l'ordre de test_hids
            labels_train = labels_train.sort_values('stay_id').reset_index(drop=True)


            y_train = labels_train.iloc[:,1]
            y_train.fillna(0, inplace=True)  # Remplacer les NaN par 0 dans Y_train
            
            output_path= os.path.join(os.environ['TMPDIR'], 'data4/output') 
            with open(os.path.join(output_path,"model_results.txt"), "a") as file:
                file.write("==================={0:2d} FOLD=====================\n".format(i))
                file.write("Positive sample of train dataset :{:.2f}  \n ".format(y_train.sum()))
                file.write('train_hids:{:.2f} \n'.format(len(train_hids)))
                file.write('test_hids: {:.2f} \n'.format(len(test_hids)))
   
            if self.oversampling:
                #print("=============OVERSAMPLING===============")
                oversample = RandomOverSampler(sampling_strategy='minority')
                train_hids=np.asarray(train_hids).reshape(-1,1)
                train_hids, y_train = oversample.fit_resample(train_hids, y_train)
                train_hids=train_hids[:,0]
                with open(os.path.join(output_path,"model_results.txt"), "a") as file:
                    file.write("=============OVERSAMPLING=============== \n")
                    file.write("Total Samples: {:.2f} \n".format(len(train_hids)))
                    file.write("Positive Samples: {:.2f} \n".format(y_train.sum()))
            
            indices_train_hids = []
            
            for id in train_hids:
                index = total_hids.index(id)
                indices_train_hids.append(index)

            #print("indices_train_hids",indices_train_hids)    
            X_train = X.iloc[indices_train_hids]
       
            # Récupérer les noms des colonnes
            columns_list = X_test.columns.tolist()
            # Sauvegarder les colonnes dans un fichier CSV
            pd.DataFrame(columns_list, columns=["Features"]).to_csv(os.path.join(output_path,'features.csv'), index=False)

            self.train_model(X_train,y_train,X_test,y_test)
    
    def train_model(self,X_train,Y_train,X_test,Y_test):
        #logits=[]
        print("===============MODEL TRAINING Xgboost===============")
        if self.model_type=='Gradient Bossting':
            model = HistGradientBoostingClassifier(categorical_features=[X_train.shape[1]-3,X_train.shape[1]-2,X_train.shape[1]-1]).fit(X_train, Y_train)
            # Enregistrer le modèle entraîné
            output_path =  os.path.join(os.environ['TMPDIR'], 'data4/output')
            joblib.dump(model, os.path.join(output_path,'trained_model'f'{self.fold}.pkl'))
            prob=model.predict_proba(X_test)
            logits=np.log2(prob[:,1]/prob[:,0])
            self.loss(prob[:,1],np.asarray(Y_test),logits,False,True,self.fold)
            self.save_output(Y_test,prob[:,1],logits)
            self.fold = self.fold +1
        
        elif self.model_type=='Logistic Regression':
            #X_train=pd.get_dummies(X_train,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            #X_test=pd.get_dummies(X_test,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            
            model = LogisticRegression().fit(X_train, Y_train) 
            output_path =  os.path.join(os.environ['TMPDIR'], 'data4/output')
            joblib.dump(model, os.path.join(output_path,'trained_model'+f'{self.fold}.pkl'))
            logits=model.predict_log_proba(X_test)
            prob=model.predict_proba(X_test)
            self.loss(prob[:,1],np.asarray(Y_test),logits[:,1],False,True,self.fold)
            self.save_outputImp(Y_test,prob[:,1],logits[:,1],model.coef_[0],X_train.columns)
            self.fold = self.fold +1

        
        elif self.model_type=='Random Forest':
            #X_train=pd.get_dummies(X_train,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            #X_test=pd.get_dummies(X_test,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            model = RandomForestClassifier().fit(X_train, Y_train)
            output_path =  os.path.join(os.environ['TMPDIR'], 'data4/output')
            joblib.dump(model, os.path.join(output_path,'trained_model'+f'{self.fold}.pkl'))
            logits=model.predict_log_proba(X_test) 
            prob=model.predict_proba(X_test)
            self.loss(prob[:,1],np.asarray(Y_test),logits[:,1],False,True,self.fold)
            self.save_outputImp(Y_test,prob[:,1],logits[:,1],model.feature_importances_,X_train.columns)
            self.fold = self.fold +1

        
        elif self.model_type=='Xgboost':
            #X_train=pd.get_dummies(X_train,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            #X_test=pd.get_dummies(X_test,prefix=['gender','ethnicity','insurance'],columns=['gender','ethnicity','insurance'])
            model = xgb.XGBClassifier(objective="binary:logistic").fit(X_train, Y_train)
            output_path =  os.path.join(os.environ['TMPDIR'], 'data4/output')
            joblib.dump(model, os.path.join(output_path,'trained_model'+f'{self.fold}.pkl'))
            #logits=model.predict_log_proba(X_test)
            #print(self.test_data['ethnicity'])
            #print(self.test_data.shape)
            #print(self.test_data.head())
            prob=model.predict_proba(X_test)
            logits=np.log2(prob[:,1]/prob[:,0])
            self.loss(prob[:,1],np.asarray(Y_test),logits,False,True,self.fold)
            self.save_outputImp(Y_test,prob[:,1],logits,model.feature_importances_,X_train.columns)
            self.fold = self.fold +1



    
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
            dyn_path = os.path.join(os.environ['TMPDIR'], 'mimic/data/csv/')
            dyn=pd.read_csv(dyn_path+str(sample)+'/dynamic.csv',header=[0,1])
            
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
            stat_path = os.path.join(os.environ['TMPDIR'], 'mimic/data/csv/')
            stat=pd.read_csv(stat_path+str(sample)+'/static.csv',header=[0,1])
            stat=stat['COND']
#             print(stat.shape)
#             print(stat.head())
            demo_path = os.path.join(os.environ['TMPDIR'], 'mimic/data/csv/')
            demo=pd.read_csv(demo_path+str(sample)+'/demo.csv',header=0)
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
        #print("X_df",X_df.shape)
        #print("y_df",y_df.shape)
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
        output_path= os.path.join(os.environ['TMPDIR'], 'data4/output') 
      
        with open(os.path.join(output_path,'outputDict'+f'{self.fold}'), 'wb') as fp:
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
        output_path= os.path.join(os.environ['TMPDIR'], 'data4/output') 
        with open(os.path.join(output_path,'outputDict'+f'{self.fold}'), 'wb') as fp:
            pickle.dump(output_df, fp)
        
        imp_df=pd.DataFrame()
        imp_df['imp']=importance
        imp_df['feature']=features
        imp_df.to_csv(os.path.join(output_path,'feature_importance'+f'{self.fold}.csv'), index=False)
                
                

