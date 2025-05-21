# 🧠 Evaluating the Potential of Generative Techniques to Improve Generalization in Federated Learning

This project explores the potential of **generative data augmentation strategies** to improve model generalization in **federated learning**, while preserving data privacy.  

In this context, **synthetic data** is generated locally from real data, without direct sharing between participating sites.

---

## 🏥 Data Used

Experiments were conducted using the **ICU module** of the **MIMIC-IV** database, segmented into five cohorts corresponding to different medical conditions:

- 🫀 Heart Failure  
- 🧽 Chronic Kidney Disease (CKD)  
- 🫁 Chronic Obstructive Pulmonary Disease (COPD)  
- ❤️ Coronary Artery Disease (CAD)  
- 🔬 Other conditions (not included in the study)

These cohorts are assumed to be distributed across **four hospitals**, each locally hosting data for one specific condition.

---

## 📁 Project Structure

- `data/` – Simulated or real data (not included)  
- `notebooks/` – Jupyter notebooks for data exploration, modeling, and results  
- `scripts/` – Scripts for data preprocessing, federation, and synthetic data generation  
- `results/` – Figures, model scores, and result summaries

---

## 🎯 Objectives

1. Data preprocessing and cleaning  
2. Training classical models locally  
3. Cross-site evaluation  
4. Implementation of federated learning with:
   - Federated Logistic Regression  
   - Federated XGBoost  
5. Synthetic data generation using **RealTabFormer**  
6. Retraining models on synthetic data  
7. Evaluating generalization performance

---

## 🧪 Models Explored

| Type                       | Model                          |
|----------------------------|--------------------------------|
| Classical                  | Logistic Regression            |
|                            | Random Forest                  |
|                            | XGBoost                        |
|                            | Gradient Boosting              |
| Federated Learning         | Federated Logistic Regression  |
|                            | Federated XGBoost              |
| Generative (Augmentation)  | RealTabFormer                  |

---

## 🔍 Results

The results compare the performance of:
- **Locally trained models**
- **Federated learning models**
- **Models trained on synthetic data**

Metrics include:
- AU-ROC (Area Under the Receiver Operating Characteristic Curve)  
- AU-PRC (Area Under the Precision-Recall Curve)

---

## 👩‍💻 Author

**Siham Si Hadj Mohand**  
Master’s in Electrical Engineering · Passionate about AI, Data Science, and Embedded Systems
[LinkedIn](www.linkedin.com/in/siham-s) | [Email](siham.sihadj@gmail.com)


---

## 📚 References

- **MIMIC-IV ICU**  
  [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)

- **RealTabFormer**  
  [https://github.com/worldbank/REaLTabFormer](https://github.com/worldbank/REaLTabFormer)  
  A. V. Solatorio and O. Dupriez, _“REaLTabFormer: Generating realistic relational and tabular data using transformers,”_  
  *arXiv preprint* [arXiv:2302.02041](https://arxiv.org/abs/2302.02041), 2023.

- **Data Processing Pipeline**  
  M. Gupta, B. M. Gallamoza, N. Cutrona, P. Dhakal, R. Poulain, and R. Beheshti,  
  _“An extensive data processing pipeline for MIMIC-IV,”_  
  In *Proceedings of the 2nd Machine Learning for Health Symposium*, PMLR, Vol. 193, 2022, pp. 311–325.

- **federated XGBoost (Flower)**
  
   Documentation: https://flower.ai/blog/2024-02-14-federated-xgboost-with-flower/
  
   Repository: https://github.com/adap/flower/tree/main/examples/xgboost-quickstart

- **federated Logistic Regression**
  
   Documentation: https://fate.readthedocs.io/en/develop/_build_temp/python/federatedml/linear_model/logistic_regression/README.html
  
   Repository: https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist

  

# 🧠 Évaluation du potentiel des techniques génératives pour améliorer la généralisation en apprentissage fédéré

Ce projet explore le potentiel des **stratégies d’augmentation de données génératives** pour améliorer la généralisation des modèles en **apprentissage fédéré**, tout en respectant la confidentialité des données.  

Dans ce contexte, des **données synthétiques** sont générées localement à partir de données réelles, sans partage direct entre les sites participants.

---

## 🏥 Données utilisées

Les expériences ont été menées à partir du module **ICU** de la base de données **MIMIC-IV**, segmenté en cinq cohortes correspondant à différentes pathologies :

- 🫀 Insuffisance cardiaque
- 🧽 Maladie rénale chronique (CKD)
- 🫁 Maladie pulmonaire obstructive chronique (COPD)
- ❤️ Maladie coronarienne (CAD)
- 🔬 Autres maladies (non incluses dans l’étude)

Ces cohortes sont supposées réparties entre **quatre hôpitaux**, chacun hébergeant localement les données d’une maladie spécifique.

---

## 📁 Structure du projet

- `data/` : Données simulées ou réelles (non incluses)
- `notebooks/` : Notebooks Jupyter pour exploration, modélisation et résultats
- `scripts/` : Scripts pour le traitement, fédération et génération de données
- `results/` : Graphiques, scores des modèles

---

## 🎯 Objectifs

1. Prétraitement et nettoyage des données
2. Entraînement de modèles classiques localement
3. Évaluation croisée entre les sites
4. Implémentation de l’apprentissage fédéré avec :
   - Régression logistique fédérée
   - XGBoost fédéré
5. Génération de données synthétiques avec **RealTabFormer**
6. Réentraînement des modèles sur données synthétiques
7. Évaluation des performances généralisées

---

## 🧪 Modèles explorés

| Type                       | Modèle                       |
|----------------------------|------------------------------|
| Classique                  | Régression Logistique        |
|                            | Random Forest                |
|                            | XGBoost                      |
|                            | Gradient Boosting            |
| Apprentissage fédéré       | Régression Logistique fédérée |
|                            | XGBoost fédéré               |
| Génératif (augmentation)   | RealTabFormer                |

---

## 🔍 Résultats

Les résultats comparent les performances :
- des modèles **entraînés localement**
- des modèles **entraînés via FL**
- des modèles **entraînés sur données synthétiques**

Les métriques incluent :
- AU-ROC
- AU-PRC


---


## 👩‍💻 Auteur

**Siham Si Hadj Mohand**  
Maîtrise en génie électrique · Passionnée par l’IA, la science des données et les systèmes embarqués  
[LinkedIn](www.linkedin.com/in/siham-s) | [Email](siham.sihadj@gmail.com)

---

## 📚 Références

- **MIMIC-IV ICU** : [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)
- **RealTabFormer** :
  [https://github.com/worldbank/REaLTabFormer](https://github.com/worldbank/REaLTabFormer)
  
  A. V. Solatorio and O. Dupriez, _“REaLTabFormer: Generating realistic relational and tabular data using transformers,”_  
  *arXiv preprint* [arXiv:2302.02041](https://arxiv.org/abs/2302.02041), 2023.
- **Pipeline de traitement utilisé** :  
  M. Gupta, B. M. Gallamoza, N. Cutrona, P. Dhakal, R. Poulain, and R. Beheshti,  
  _“An extensive data processing pipeline for MIMIC-IV,”_  
  in *Proceedings of the 2nd Machine Learning for Health Symposium*, PMLR, Vol. 193, 2022, pp. 311–325.

- **federated XGBoost (Flower)**
  
   Documentation: https://flower.ai/blog/2024-02-14-federated-xgboost-with-flower/
  
   Repository: https://github.com/adap/flower/tree/main/examples/xgboost-quickstart

- **federated Logistic Regression**
  
   Documentation: https://fate.readthedocs.io/en/develop/_build_temp/python/federatedml/linear_model/logistic_regression/README.html
  
   Repository: https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist


