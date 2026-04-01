# 📋 Rapport Synthétique du Modèle
## Prédiction du Trafic Urbain — Smart City
### Interstate 94, Minnesota, USA

---

## 1. OBJECTIF

**Problématique choisie :**
Estimer le volume de trafic routier (en véhicules/heure) sur l'Interstate 94
à partir de l'heure de la journée, du jour de la semaine et des conditions
météorologiques.

**Contexte Smart City :**
Dans le cadre d'une ville intelligente, la prédiction du trafic permet de :
- Adapter les feux de signalisation en temps réel
- Informer les usagers des axes surchargés
- Optimiser les déploiements de transports en commun
- Planifier les interventions de voirie (travaux, incidents)

**Variable cible :** `traffic_volume` — nombre de véhicules par heure

---

## 2. MÉTHODE

### 2.1 Dataset
| Attribut | Valeur |
|---|---|
| Source | UCI Machine Learning Repository |
| Observations | 48 204 enregistrements horaires |
| Période | Octobre 2012 – Septembre 2018 |
| Localisation | Interstate 94, Minneapolis–Saint Paul, Minnesota |

### 2.2 Variables utilisées (features)

**Variables temporelles :**
- `hour` — heure de la journée (0–23)
- `day_of_week` — jour de la semaine (0=Lundi, 6=Dimanche)
- `month` — mois (1–12)
- `year` — année
- `is_weekend` — 1 si samedi ou dimanche, 0 sinon
- `is_holiday` — 1 si jour férié, 0 sinon

**Variables météorologiques :**
- `temp` — température en Kelvin (convertie depuis °C)
- `rain_1h` — précipitations pluie en mm/heure
- `snow_1h` — précipitations neige en mm/heure
- `clouds_all` — couverture nuageuse en %
- `weather_encoded` — condition météo encodée (11 catégories)

**Moyennes glissantes (rolling features) :**
- `traffic_rolling_3h` — moyenne du trafic sur les 3 dernières heures
- `traffic_rolling_24h` — moyenne du trafic sur les 24 dernières heures

### 2.3 Prétraitement
- Tri chronologique du dataset
- Extraction des features temporelles depuis `date_time`
- Encodage de `weather_main` (LabelEncoder) : 11 conditions → entiers
- Création des features binaires `is_weekend` et `is_holiday`
- Calcul des moyennes glissantes avec `.rolling()` de pandas
- Découpage 80% entraînement / 20% test (random_state=42)

### 2.4 Modèles entraînés
Trois modèles ont été comparés, du plus simple au plus complexe :

1. **Régression Linéaire** — modèle de référence (baseline)
2. **Gradient Boosting Regressor** — 100 arbres séquentiels, lr=0.1
3. **Random Forest Regressor** — 100 arbres parallèles ← **retenu**

---

## 3. RÉSULTATS

### 3.1 Performances comparées

| Modèle | MAE (véh/h) | RMSE (véh/h) | R² |
|---|---|---|---|
| Régression Linéaire | 477 | 672 | 0.886 |
| Gradient Boosting | 211 | 314 | 0.975 |
| **Random Forest ⭐** | **129** | **224** | **0.987** |

*MAE = Erreur Absolue Moyenne | RMSE = Erreur Quadratique Moyenne | R² = Coefficient de détermination*

### 3.2 Interprétation du meilleur modèle (Random Forest)

- **R² = 0.987** → le modèle explique 98.7% de la variance du trafic
- **MAE = 129 véhicules/heure** → erreur moyenne de ±129 véhicules sur les prédictions
- **RMSE = 224 véhicules/heure** → les grandes erreurs restent rares
- Validation croisée 5-fold : R² moyen stable, faible écart-type → bonne généralisation

### 3.3 Importance des variables

| Rang | Variable | Importance |
|---|---|---|
| 1 | Moyenne glissante 3h | **78.8%** |
| 2 | Heure de la journée | 19.7% |
| 3 | Jour de la semaine | 0.31% |
| 4 | Température | 0.28% |
| 5 | Mois | 0.14% |
| … | Météo, pluie, neige, etc. | < 0.2% chacun |

**Observation clé :** la moyenne glissante 3h (78.8%) confirme que le trafic
présente une forte autocorrélation temporelle — la meilleure prédiction du
trafic présent est le trafic récent.

### 3.4 Gain apporté par les rolling features

| Modèle | Sans rolling | Avec rolling | Gain |
|---|---|---|---|
| Random Forest | R²=0.956 | R²=0.987 | **+3.1 pts** |

---

## 4. LIMITES

### Limites du modèle
- **Accidents et incidents** : le modèle ne connaît pas les événements imprévus
  qui perturbent le trafic (accidents, pannes, événements sportifs)
- **Travaux routiers** : les déviations et réductions de voies ne sont pas modélisées
- **Dépendance aux rolling features** : la prédiction nécessite de connaître le
  trafic des dernières heures — difficile en prédiction très longue portée
- **Généralisation géographique** : le modèle a été entraîné sur une seule
  autoroute américaine ; les performances seraient inférieures sur d'autres réseaux

### Limites des données
- Période 2012–2018 : le comportement post-COVID (télétravail) n'est pas couvert
- Fréquence horaire : pas de granularité inférieure à 1h
- Pas d'information sur la capacité routière réelle ni sur le nombre de voies

### Pistes d'amélioration
- Intégrer des données d'incidents en temps réel (API accidents)
- Utiliser un modèle de série temporelle (LSTM, Prophet) pour la prédiction longue portée
- Ajouter des features cycliques (sin/cos de l'heure) pour encoder la périodicité
- Tester XGBoost / LightGBM pour potentiellement améliorer encore les performances

---

## 5. CONCLUSION

Le modèle **Random Forest** entraîné sur 48 204 observations horaires atteint
un **R² de 0.987** avec une erreur moyenne de **±129 véhicules/heure**.

L'ajout des **moyennes glissantes** (rolling features) s'est révélé décisif :
le trafic présent est fortement corrélé au trafic récent, ce qui représente
le signal prédictif le plus fort.

Ce modèle est opérationnel pour une intégration dans un système Smart City
de gestion du trafic en temps réel, sous réserve d'un flux de données horaires
en continu alimentant les features de type rolling.

---

*Rapport généré dans le cadre du projet Data Scientist — Smart City*
*Modèle : Random Forest Regressor | Dataset : UCI Metro Interstate Traffic Volume*
