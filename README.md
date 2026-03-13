                                                        🏥 Insurance Insight Pro

Insurance Insight Pro est une plateforme interactive d'analyse de données et d'aide à la décision appliquée aux coûts de l'assurance maladie aux États-Unis. Inspirée par l'esthétique minimaliste et épurée de Claude Opus, cette application combine exploration de données (EDA), détection de biais algorithmiques et évaluation de modèles de Machine Learning.

🚀 Aperçu du Projet

L'objectif de cette application est de décrypter les mécanismes de tarification des primes d'assurance et d'évaluer l'équité du système vis-à-vis des différents groupes démographiques. Elle s'articule autour de quatre axes majeurs :

Accueil & Contexte : Présentation du dataset et des enjeux socio-économiques.

Exploration Interactive : Analyse visuelle des facteurs de coûts avec un assistant virtuel pédagogique.

Audit d'Équité : Détection de biais statistiques (Statistical Parity & Disparate Impact) sur les attributs sensibles.

Performance Prédictive : Analyse approfondie d'un modèle de forêt aléatoire (Random Forest) et de l'importance des variables.

🛠️ Stack Technique

Framework : Streamlit

Analyse de données : Pandas, NumPy

Visualisation : Plotly Express (Interactivité haute fidélité)

Machine Learning : Scikit-Learn (Random Forest Regressor)

Design : CSS personnalisé (Typographies Inter & Ibarra Real Nova)

📋 Structure de l'Application

1. Accueil

Présentation détaillée du dataset "Medical Cost Personal Datasets". Cette section inclut :

La problématique du projet.

Des KPIs sur la santé des données (taux de complétude, statistiques de la cible).

Un dictionnaire des variables pour une compréhension immédiate du métier.

2. Exploration des Données

Visualisations interactives avec un assistant de chat qui se déroule en temps réel pour expliquer les insights :

Distribution des charges avec analyse de l'asymétrie.

Analyse par segments (Fumeurs vs Non-fumeurs, impact de l'IMC).

Corrélations multi-dimensionnelles via scatter plots.

3. Détection de Biais

Un module dédié à l'éthique des données permettant de tester l'impact d'attributs sensibles (Sexe, Région) sur la probabilité d'avoir des coûts élevés. Les métriques calculées incluent :

Statistical Parity Difference

Disparate Impact Ratio

4. Performance du Modèle

Analyse du modèle de prédiction final (Random Forest) :

Visualisation du score $R^2$ et de la MAE.

Graphique de Feature Importance pour comprendre quels facteurs dictent réellement le prix.

💻 Installation Locale

Pour exécuter cette application sur votre machine, suivez ces étapes :

Cloner le dépôt :

git clone [https://github.com/votre-utilisateur/insurance-insight-pro.git](https://github.com/votre-utilisateur/insurance-insight-pro.git)
cd insurance-insight-pro


Installer les dépendances :

pip install -r requirements.txt


Lancer l'application :

streamlit run app.py


📦 Fichiers du Dépôt

app.py : Code principal de l'interface Streamlit.

insurance_cleaned.csv : Dataset traité et nettoyé.

metadata.json : Métadonnées structurées du projet.

insurance_model_v1.pkl : Poids du modèle Random Forest entraîné.

requirements.txt : Liste des bibliothèques nécessaires au déploiement.

Projet réalisé dans le cadre du module "Streamlit & Data Science".