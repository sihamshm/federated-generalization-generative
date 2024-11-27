# projet-d-essai
.1 Problématique
La classification des données biomédicales à ressources limitées est un défi important pour l’IA
dans le domaine de la santé [1; 2]. Ce défi se manifeste lorsque les ressources disponibles pour
entraîner les modèles sont limitées, principalement en raison de deux facteurs :
-Manque de données [2] : Les données biomédicales sont souvent disponibles en quantités
limitées, par exemple dans des groupes démographiques spécifiques ou pour des conditions
médicales moins courantes.
-Problèmes de confidentialité [3] : Les données biomédicales sont sensibles et soumises à des
réglementations strictes en matière de confidentialité, ce qui limite la possibilité de les partager
et de les utiliser pour l’apprentissage automatique.
Le manque de données a un impact négatif sur la performance des modèles de classification
[4] : l’entraînement sur des petits jeux de données peut mener au sur-apprentissage, ce qui nuit
à la précision et à la capacité de généralisation des modèles sur de nouvelles données n’ayant
pas servi à l’apprentissage.
I.2 Objectives
L’objectif de ce projet est d’explorer le potentiel des stratégies d’augmentation de données
pour améliorer la généralisation du modèle en apprentissage fédéré tout en respectant la confi-
dentialité des données. Cela nécessite la génération des données synthétiques à partir des
données réelles provenant des différents sites où le partage de données réelles est impossible.
