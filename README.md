<<<<<<< HEAD
# 🚦 Prédiction du Trafic Urbain — Smart City
## Guide de démarrage complet

---

## 📁 Structure du projet

```
traffic_project/
├── Prediction_Trafic_Urbain.ipynb   ← Notebook Jupyter complet (10 sections)
├── Rapport_Synthetique_Modele.md    ← Rapport (objectif, méthode, résultats, limites)
├── app.py                           ← Interface Streamlit (5 onglets)
├── Metro_Interstate_Traffic_Volume.csv  ← Dataset
├── model_rf.pkl                     ← Random Forest entraîné
├── model_gb.pkl                     ← Gradient Boosting entraîné
├── model_meta.json                  ← Métadonnées & résultats
├── requirements.txt                 ← Dépendances Python
└── README.md                        ← Ce fichier
```

---

## 🏆 Résultats des modèles

| Modèle             | MAE     | RMSE    | R²     |
|--------------------|---------|---------|--------|
| Régression Linéaire| 477 v/h | 672 v/h | 0.886  |
| Gradient Boosting  | 211 v/h | 314 v/h | 0.975  |
| **Random Forest ⭐**| **129 v/h** | **224 v/h** | **0.987** |

---

## ▶️ Installation et lancement

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Lancer le notebook Jupyter
```bash
jupyter notebook Prediction_Trafic_Urbain.ipynb
```

### 3. Lancer l'interface Streamlit
```bash
streamlit run app.py
```
L'application s'ouvre sur `http://localhost:8501`

---

## 📋 Contenu du Notebook (10 sections)

1. Importation des bibliothèques
2. Chargement et exploration du dataset
3. Analyse Exploratoire (EDA) — histogrammes, heatmaps, corrélations
4. **Feature Engineering** — moyennes glissantes, encodage, variables dérivées
5. Prétraitement — split train/test 80/20
6. Entraînement de 3 modèles ML
7. Évaluation — MAE, RMSE, R², résidus, validation croisée
8. Importance des variables — feature importance graphique
9. Prédictions — fonction réutilisable + prévision 4 scénarios 24h
10. Sauvegarde des modèles (.pkl)

---

## 🌐 Interface Streamlit (5 onglets)

| Onglet | Contenu |
|--------|---------|
| 🔮 Prédiction | Résultat temps réel + prévision 24h |
| 📊 Exploration | Stats, graphiques trafic par heure/jour/météo |
| 📈 Performance | Comparaison des 3 modèles |
| 🧠 Importance | Feature importance graphique |
| 📋 Rapport | Rapport synthétique intégré |
=======
# Projet_fil_conducteur_africa_Techup
Ce projet est un projet de fin d’apprentissage
>>>>>>> 34070135f0566ee1c55bb77b303f2d3b08457175
