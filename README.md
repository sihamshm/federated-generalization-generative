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
- `results/` : Graphiques, matrices de confusion, scores des modèles

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
- Accuracy


---

## 💡 Perspectives

- Étendre à d’autres générateurs (CTGAN, TVAE…)
- Étudier l’impact de l’augmentation sur des sites faiblement représentés
- Intégrer la détection de biais dans les données générées

---

## 👩‍💻 Auteur

**Ton Prénom NOM**  
Étudiante en génie électrique · Passionnée par l’IA, le fédéré et les données médicales  
[LinkedIn](https://linkedin.com/in/tonprofil) | [Email](mailto:ton@email.com)

---

## 📚 Références

- MIMIC-IV ICU : [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)
- RealTabFormer : [Article officiel ou GitHub du modèle]
- [15] Référence du pipeline de traitement utilisé (à compléter)

