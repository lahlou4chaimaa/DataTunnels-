# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 22:22:55 2025

@author: achra
"""

# -*- coding: utf-8 -*-
"""
AFC sur le tableau tunnels × 9 critères structurels
"""

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Chargement des données tunnels ---
file_path = 'tunnels_data_100.csv'  # Modifier selon votre chemin
data_tunnels = pd.read_csv(file_path)

# Vérification des données
print("Dimensions des données :", data_tunnels.shape)
print("\nPremières lignes :")
print(data_tunnels.head())
print("\nColonnes :")
print(data_tunnels.columns)

# --- Nettoyage et préparation ---
# Nettoyage des noms de colonnes
data_tunnels.columns = ['ID', 'Nom du Tunnel', 'Pression terrain (MPa)', 'Convergence (mm)', 
                        'Humidité (%)', 'Température (°C)', 'Vibrations (Hz)', 
                        'Fissures (mm)', 'Ventilation (m/s)', 'Durée exploitation (ans)']

# Nettoyage des colonnes numériques
numeric_cols = ['Pression terrain (MPa)', 'Convergence (mm)', 'Humidité (%)', 
                'Température (°C)', 'Vibrations (Hz)', 'Fissures (mm)', 
                'Ventilation (m/s)', 'Durée exploitation (ans)']

for col in numeric_cols:
    data_tunnels[col] = pd.to_numeric(data_tunnels[col], errors='coerce')

# Suppression des valeurs manquantes
data_tunnels_clean = data_tunnels.dropna()

# Vérification
print(f"\nDonnées nettoyées : {data_tunnels_clean.shape[0]} tunnels")
print(data_tunnels_clean.head())

X = data_tunnels_clean[numeric_cols]

# ----- Partie 1 : Statistiques descriptives et matrices -----
print("\n=== Partie 1 : Statistiques descriptives et matrices ===\n")

# 1 - Moyenne, écart-type et interprétation
moyennes = X.mean()
ecarts_type = X.std()
print("1 - Moyenne et Écart-Type par critère :")
for c in numeric_cols:
    print(f"{c} : Moyenne = {moyennes[c]:.2f}, Écart-type = {ecarts_type[c]:.2f} | "
          f"Interprétation : La moyenne exprime la tendance centrale, l'écart-type la dispersion.")

# 2 - Matrice centrée réduite
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, 
                           index=data_tunnels_clean['Nom du Tunnel'])

print("\n2 - Matrice centrée réduite (extrait) :")
print(X_scaled_df.head())

# Visualisation heatmap (divisée en 4 parties)
n_tunnels_par_figure = 25
kwargs_heatmap = {
    'annot': True,
    'fmt': '.2f',
    'cmap': 'RdBu_r',
    'center': 0,
    'linewidths': 0.8,
    'linecolor': 'gray',
    'cbar_kws': {'shrink': 0.8},
    'annot_kws': {'size': 6}
}

for i in range(4):
    debut = i * n_tunnels_par_figure
    fin = min((i + 1) * n_tunnels_par_figure, len(X_scaled_df))
    
    if debut >= len(X_scaled_df):
        break
        
    plt.figure(figsize=(14, 7))
    sns.heatmap(X_scaled_df.iloc[debut:fin], **kwargs_heatmap)
    plt.title(f"Matrice standardisée - Tunnels {debut+1}-{fin}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 3 - Matrice de corrélation
corr_df = pd.DataFrame(np.corrcoef(X_scaled.T), index=numeric_cols, columns=numeric_cols)
print("\n3 - Matrice de corrélation :")
print(corr_df.round(2))

plt.figure(figsize=(10, 7))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()

# 4 - Valeurs propres et inertie
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

valeurs_propres = pca.explained_variance_
inertie_expliquee = pca.explained_variance_ratio_
inertie_cumulee = np.cumsum(inertie_expliquee)

print("\n4 - Valeurs propres et inertie expliquée :")
for i, val in enumerate(valeurs_propres, 1):
    print(f"Composante {i} : {val:.4f} | Inertie expliquée : {inertie_expliquee[i-1]:.4f} "
          f"({inertie_expliquee[i-1]*100:.2f}%) - Cumulée : {inertie_cumulee[i-1]*100:.2f}%")

plt.figure(figsize=(8,5))
plt.plot(range(1, len(inertie_expliquee)+1), inertie_expliquee*100, 
         marker='o', label='Inertie par composante')
plt.plot(range(1, len(inertie_cumulee)+1), inertie_cumulee*100, 
         marker='s', label='Inertie cumulée')
plt.title("Inertie expliquée par les composantes principales")
plt.xlabel("Composantes principales")
plt.ylabel("Pourcentage d'inertie")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ----- Partie 2 : Analyse en Composantes Principales -----
print("\n=== Partie 2 : Analyse en Composantes Principales ===\n")

# 1 - Composantes principales des individus et variables
individus_pca = pd.DataFrame(X_pca, 
                            columns=[f"Composante_{i+1}" for i in range(X_pca.shape[1])], 
                            index=data_tunnels_clean['Nom du Tunnel'])
variables_pca = pd.DataFrame(pca.components_, 
                            columns=numeric_cols,
                            index=[f"Composante_{i+1}" for i in range(len(numeric_cols))])

print("1 - Composantes principales des individus (extrait) :")
print(individus_pca.head())
print("\n1 - Composantes principales des variables :")
print(variables_pca)

# Visualisation des composantes des individus
for i in range(4):
    debut = i * 25
    fin = min((i + 1) * 25, len(individus_pca))
    
    if debut >= len(individus_pca):
        break
        
    plt.figure(figsize=(14, 7))
    sns.heatmap(individus_pca.iloc[debut:fin], 
                annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.8, linecolor='gray',
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 6})
    plt.title(f"Composantes principales - Tunnels {debut+1}-{fin}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Visualisation des composantes des variables
plt.figure(figsize=(10, 7))
sns.heatmap(variables_pca.iloc[:5],  # 5 premières composantes
            annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.8, linecolor='gray',
            cbar_kws={'shrink': 0.8})
plt.title("Composantes principales des variables")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 2 - Plan factoriel des individus
plt.figure(figsize=(10, 8))
plt.scatter(individus_pca['Composante_1'], individus_pca['Composante_2'], alpha=0.7)
for i, txt in enumerate(individus_pca.index[:20]):  # Limité à 20 pour lisibilité
    plt.annotate(txt[:15],  # Nom tronqué
                (individus_pca['Composante_1'][i], individus_pca['Composante_2'][i]), 
                fontsize=8, alpha=0.7)
plt.title("Plan factoriel des tunnels (Composantes 1 et 2)")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3 - Plan factoriel des variables
plt.figure(figsize=(10, 8))
for i, var in enumerate(numeric_cols):
    plt.arrow(0, 0, 
              variables_pca.loc['Composante_1', var], 
              variables_pca.loc['Composante_2', var], 
              head_width=0.03, head_length=0.03, color='red')
    plt.text(variables_pca.loc['Composante_1', var]*1.15, 
             variables_pca.loc['Composante_2', var]*1.15, 
             var, color='blue', fontsize=9)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axhline(0, color='grey', linewidth=1)
plt.axvline(0, color='grey', linewidth=1)
plt.title("Plan factoriel des variables (Composantes 1 et 2)")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4 - Qualité de représentation
coord_carres = individus_pca[['Composante_1', 'Composante_2']]**2
qualite_rep = pd.DataFrame()
qualite_rep['Qualité Axe 1'] = coord_carres['Composante_1'] / np.sum(coord_carres['Composante_1'])
qualite_rep['Qualité Axe 2'] = coord_carres['Composante_2'] / np.sum(coord_carres['Composante_2'])
qualite_rep['Qualité Plan factoriel'] = qualite_rep['Qualité Axe 1'] + qualite_rep['Qualité Axe 2']

print("\n4 - Qualité de représentation des tunnels (extrait) :")
print(qualite_rep.head().round(3))

# 5 - Contribution des individus et variables
contrib_ind = (individus_pca[['Composante_1', 'Composante_2']]**2).divide(
    individus_pca[['Composante_1', 'Composante_2']]**2.sum(axis=0), axis=1) * 100

contrib_var = (variables_pca.loc[['Composante_1', 'Composante_2'], numeric_cols]**2).divide(
    variables_pca.loc[['Composante_1', 'Composante_2'], numeric_cols]**2.sum(axis=1), axis=0) * 100

print("\n5 - Contribution des tunnels (extrait) :")
print(contrib_ind.head().round(2))
print("\n5 - Contribution des variables :")
print(contrib_var.round(2))

# ----- Partie 3 : Classification avec K-Means -----
from sklearn.cluster import KMeans

print("\n=== Partie 3 : Classification des tunnels ===\n")

# K-Means avec K=3, 4, 5
for k in [3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_label = f'Cluster_K{k}'
    individus_pca[cluster_label] = kmeans.fit_predict(individus_pca[['Composante_1', 'Composante_2']])
    
    print(f"\nK-Means (K={k}) :")
    print(f"Répartition : {individus_pca[cluster_label].value_counts().sort_index().to_dict()}")
    
    # Visualisation
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(individus_pca['Composante_1'], individus_pca['Composante_2'],
                         c=individus_pca[cluster_label], cmap='Set2', s=50, alpha=0.7)
    plt.title(f"Classification des tunnels (K={k})")
    plt.xlabel("Composante 1")
    plt.ylabel("Composante 2")
    plt.colorbar(scatter, label=f'Cluster K={k}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Profils moyens par cluster
    X_with_clusters = X.copy()
    X_with_clusters[cluster_label] = individus_pca[cluster_label]
    cluster_profiles = X_with_clusters.groupby(cluster_label).mean()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_profiles, annot=True, fmt='.2f', cmap='Reds', 
                linewidths=0.5, linecolor='white')
    plt.title(f"Profils moyens des clusters (K={k})")
    plt.tight_layout()
    plt.show()

# ----- Partie 4 : Analyse approfondie -----
print("\n=== Partie 4 : Analyse approfondie ===\n")

# 1. Corrélations avec la durée d'exploitation
print("Corrélations avec la durée d'exploitation :")
for col in numeric_cols:
    if col != 'Durée exploitation (ans)':
        corr = np.corrcoef(data_tunnels_clean[col], 
                          data_tunnels_clean['Durée exploitation (ans)'])[0,1]
        print(f"{col} vs Durée exploitation : {corr:.3f}")

# 2. Tunnels à risque (critères élevés)
seuil_risque = 0.8  # 80ème percentile
risques = {}
for col in ['Convergence (mm)', 'Fissures (mm)', 'Vibrations (Hz)']:
    seuil = np.percentile(data_tunnels_clean[col], 80)
    tunnels_risque = data_tunnels_clean[data_tunnels_clean[col] > seuil]['Nom du Tunnel'].tolist()
    risques[col] = tunnels_risque
    print(f"\nTunnels avec {col} élevé (> {seuil:.1f}) : {len(tunnels_risque)}")
    for tunnel in tunnels_risque[:5]:  # Premier 5
        print(f"  - {tunnel}")

# 3. Visualisation des caractéristiques par cluster
cluster_selected = 'Cluster_K4'  # Choisir un clustering
for cluster_num in sorted(individus_pca[cluster_selected].unique()):
    tunnels_cluster = individus_pca[individus_pca[cluster_selected] == cluster_num].index
    data_cluster = data_tunnels_clean[data_tunnels_clean['Nom du Tunnel'].isin(tunnels_cluster)]
    
    print(f"\nCluster {cluster_num} ({len(tunnels_cluster)} tunnels) :")
    print(f"Moyenne durée exploitation : {data_cluster['Durée exploitation (ans)'].mean():.1f} ans")
    print(f"Moyenne convergence : {data_cluster['Convergence (mm)'].mean():.1f} mm")
    print(f"Moyenne fissures : {data_cluster['Fissures (mm)'].mean():.1f} mm")

# 4. Export des résultats
resultats = data_tunnels_clean.copy()
resultats = resultats.merge(individus_pca[['Composante_1', 'Composante_2', 'Cluster_K3', 'Cluster_K4']],
                           left_on='Nom du Tunnel', right_index=True, how='left')

print("\n=== Résultats finaux (extrait) ===")
print(resultats[['Nom du Tunnel', 'Durée exploitation (ans)', 
                'Convergence (mm)', 'Fissures (mm)', 
                'Cluster_K3', 'Cluster_K4']].head())

# Visualisation du tableau complet
cols_viz = numeric_cols + ['Composante_1', 'Composante_2', 'Cluster_K4']
resultats_viz = resultats.set_index('Nom du Tunnel')[cols_viz]

# Visualisation par segments
for i in range(4):
    debut = i * 25
    fin = min((i + 1) * 25, len(resultats_viz))
    
    if debut >= len(resultats_viz):
        break
        
    plt.figure(figsize=(16, 10))
    sns.heatmap(resultats_viz.iloc[debut:fin], annot=True, fmt='.2f', 
                cmap='YlOrBr', linewidths=0.5, linecolor='grey',
                cbar_kws={'shrink': 0.8})
    plt.title(f"Caractéristiques des tunnels {debut+1}-{fin}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

print("\n=== Analyse terminée ===")
print(f"Nombre total de tunnels analysés : {len(data_tunnels_clean)}")
print(f"Variables analysées : {', '.join(numeric_cols)}")