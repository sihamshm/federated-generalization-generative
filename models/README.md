# ============================================================================
# Ce script est une version modifiée d’un des composants du pipeline initial publié par :
# M. Gupta, B. M. Gallamoza, N. Cutrona, P. Dhakal, R. Poulain, and R. Beheshti,
# “An extensive data processing pipeline for MIMIC-IV,” in Proceedings of the
# 2nd Machine Learning for Health symposium, PMLR 193:311–325, 2022.
# Dépôt original : https://github.com/healthylaife/MIMIC-IV-Data-Pipeline
#
# Licence d’origine : MIT License
#
# Modifications apportées par : [Siham Si Hadj Mohand]
# Contexte : Projet de recherche sur l’évaluation du potentiel des techniques génératives pour améliorer la généralisation en apprentissage fédéré.
# Description des modifications : [-Séparation de l’étape de prétraitement des données de celle de l’entraînement des modèles -Réduction de la redondance dans le traitement des données en modifiant le pipeline original qui récupérait les données à chaque itération. Désormais, les données sont extraites et enregistrées une seule fois dans un DataFrame avant les boucles de traitement.]
# ============================================================================
