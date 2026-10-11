import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate 

# Configuration du style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Charger les données
data = """ID,Nom du Tunnel,Pression terrain (MPa),Convergence (mm),HumiditÃ© (%),TempÃ©rature (Â°C),Vibrations (Hz),Fissures (mm),Ventilation (m/s),DurÃ©e exploitation (ans)
1,"Tunnel Tanger-TÃ©touan",2.07,0.69,35.20,12.73,6.63,3.07,7.20,1
2,"Tunnel Rabat-SalÃ©",0.13,16.45,67.73,13.24,11.57,10.07,2.47,54
3,"Tunnel Casablanca Nord",2.07,23.97,97.39,16.20,8.70,2.76,8.07,27
4,"Tunnel Casablanca Sud",2.93,14.62,66.14,25.93,4.79,14.09,6.35,93
5,"Tunnel FÃ¨s-MeknÃ¨s",2.61,5.68,90.03,33.79,11.28,8.14,2.60,6
6,"Tunnel Marrakech-Safi",1.33,12.59,36.97,22.06,5.32,11.76,6.23,56
7,"Tunnel Agadir-Essaouira",3.62,4.47,71.15,26.17,7.59,1.23,8.45,68
8,"Tunnel Oujda Oriental",0.45,43.06,64.18,10.26,1.73,8.99,6.81,89
9,"Tunnel Kenitra-Larache",4.85,4.91,72.84,11.06,20.46,0.19,6.24,26
10,"Tunnel Taza-Al Hoceima",0.78,36.91,83.56,25.12,23.35,14.49,6.86,63
11,"Tunnel Ifrane-Azrou",0.33,45.26,23.51,24.33,16.26,2.61,8.42,34
12,"Tunnel Midelt-Errachidia",4.30,17.60,38.35,29.89,12.46,3.07,1.43,70
13,"Tunnel Ouarzazate-Zagora",1.09,28.62,36.72,34.43,10.40,13.85,4.89,9
14,"Tunnel Beni Mellal",3.91,28.94,1.36,17.66,13.19,9.01,0.77,75
15,"Tunnel Khouribga-Settat",1.34,17.32,66.81,20.19,14.63,10.40,5.14,21
16,"Tunnel Safi-El Jadida",1.22,46.24,63.96,30.78,2.09,4.32,5.88,41
17,"Tunnel Nador-Berkane",2.87,10.76,31.06,36.91,0.86,3.18,7.20,46
18,"Tunnel Guelmim-Tan Tan",2.64,20.75,91.58,28.64,10.33,12.17,0.70,13
19,"Tunnel Laayoune-Dakhla",2.83,6.47,98.60,25.84,6.74,5.91,3.17,91
20,"Tunnel TÃ©touan-Chefchaouen",3.86,7.40,45.94,23.76,12.61,5.14,1.04,81
21,"Tunnel Mohammedia-Benslimane",4.87,44.56,45.97,31.91,7.88,13.38,1.45,80
22,"Tunnel Boujdour-Smara",2.58,25.95,73.98,31.27,7.68,8.60,0.09,93
23,"Tunnel Tiznit-Sidi Ifni",3.55,35.53,79.35,24.87,6.76,4.87,5.61,10
24,"Tunnel Assa-Zag",4.69,5.39,63.32,6.65,15.61,4.04,0.72,53
25,"Tunnel Tinghir-Boumalne",2.16,21.35,23.43,23.73,13.25,2.41,9.98,80
26,"Tunnel Azilal-Demnate",3.62,11.80,64.61,16.90,16.65,7.38,4.26,71
27,"Tunnel Imouzzer-Kandar",3.76,19.94,76.98,34.55,16.87,4.15,1.83,67
28,"Tunnel Sefrou-Boulemane",3.21,9.35,80.29,14.90,6.79,7.33,6.16,86
29,"Tunnel Taourirt-Jerada",0.72,2.88,35.05,11.52,0.62,2.67,0.32,94
30,"Tunnel Tarfaya-Laayoune",3.77,4.85,20.26,22.26,19.97,3.69,8.14,55
31,"Tunnel Khenifra-Mrirt",2.41,6.61,77.90,16.43,21.62,13.36,1.97,43
32,"Tunnel Al Haouz",1.53,25.45,24.01,20.40,17.00,10.74,7.11,12
33,"Tunnel DrÃ¢a-Tafilalet",1.36,26.80,26.50,16.45,8.26,7.69,2.96,51
34,"Tunnel Souss-Massa",1.43,29.79,49.57,5.01,16.09,10.50,0.74,86
35,"Tunnel Gharb-Chrarda",2.70,46.16,64.82,20.21,24.12,6.58,7.99,62
36,"Tunnel Chaouia-Ouardigha",3.22,2.45,29.04,28.79,15.07,6.12,9.63,88
37,"Tunnel Doukkala-Abda",1.92,45.57,28.66,27.29,23.51,3.75,5.81,47
38,"Tunnel Tadla-Azilal",0.92,31.69,90.82,22.61,5.31,10.20,6.26,74
39,"Tunnel Tanger-Asilah",1.00,48.48,61.22,35.95,23.73,4.00,1.49,99
40,"Tunnel Larache-Ksar el Kebir",0.57,20.61,48.48,24.94,2.23,13.11,0.51,60
41,"Tunnel Sidi Kacem-Sidi Slimane",2.83,48.71,35.34,39.28,18.91,11.60,5.28,82
42,"Tunnel Khemisset-Tiflet",1.37,31.54,56.35,14.58,17.57,6.33,1.87,16
43,"Tunnel Berrechid-Bouznika",4.03,21.57,42.38,9.62,10.58,13.59,8.17,58
44,"Tunnel Settat-Ben Ahmed",0.12,13.86,31.71,11.86,23.24,0.78,8.38,19
45,"Tunnel Khouribga-Oued Zem",1.01,18.30,47.78,11.37,4.94,3.10,4.10,75
46,"Tunnel Youssoufia-Safi",3.97,15.12,43.69,28.01,6.90,10.17,2.13,75
47,"Tunnel Essaouira-Chichaoua",4.45,43.84,13.26,22.04,1.66,12.74,8.88,19
48,"Tunnel Kelaa des Sraghna",1.57,25.44,68.84,37.97,19.15,14.31,1.51,7
49,"Tunnel Marrakech-Tahannaout",2.06,17.94,85.74,21.77,24.51,7.87,5.71,43
50,"Tunnel Ourika-Setti Fatma",2.47,5.29,13.33,11.48,12.48,7.47,0.12,16
51,"Tunnel Taroudant-Oulad Teima",4.72,19.61,52.86,36.05,18.55,14.97,7.45,48
52,"Tunnel Agadir-Inezgane",4.97,46.68,1.86,37.57,21.75,11.67,2.27,98
53,"Tunnel Chtouka-Ait Baha",0.87,14.90,65.26,26.85,14.04,4.98,5.44,43
54,"Tunnel Biougra-Massa",2.13,17.48,33.18,17.28,11.47,10.08,5.23,29
55,"Tunnel Guelmim-Assa",4.35,18.01,25.39,17.07,6.95,11.59,5.11,70
56,"Tunnel Tan Tan-Akhfenir",1.31,39.32,2.36,9.80,14.67,10.27,8.16,33
57,"Tunnel Boujdour-Laayoune",0.47,35.73,43.13,28.32,6.24,4.62,8.20,84
58,"Tunnel Dakhla-Aousserd",1.89,48.60,89.66,6.94,13.45,12.62,7.59,94
59,"Tunnel Figuig-Bouarfa",0.45,45.22,57.99,7.02,22.72,13.34,1.62,19
60,"Tunnel Guercif-Taourirt",1.40,41.01,16.04,10.84,11.20,3.62,9.69,46
61,"Tunnel Taza-Aknoul",4.24,5.12,28.00,18.85,24.58,4.76,3.03,2
62,"Tunnel Al Hoceima-Ajdir",4.96,49.27,65.47,30.16,21.50,4.18,9.16,34
63,"Tunnel Nador-Selouane",1.58,50.00,85.25,11.52,9.68,12.05,3.03,49
64,"Tunnel Driouch-Midar",3.07,4.23,73.75,29.46,23.67,8.01,4.76,31
65,"Tunnel Berkane-Ahfir",1.50,18.13,58.14,20.90,19.36,2.80,4.11,12
66,"Tunnel Oujda-Ain Beni Mathar",2.70,29.36,84.09,7.01,10.00,4.93,1.56,26
67,"Tunnel Jerada-Gafait",0.05,12.60,60.24,15.44,19.06,12.38,6.13,10
68,"Tunnel Bouarfa-Tendrara",1.76,18.47,69.32,25.89,24.62,14.20,9.79,88
69,"Tunnel Midelt-Boumia",1.41,45.75,45.89,8.98,20.51,2.26,8.60,11
70,"Tunnel Rich-Goulmima",1.84,46.89,20.81,23.46,19.08,7.85,0.46,76
71,"Tunnel Errachidia-Erfoud",2.38,16.97,32.35,21.81,20.63,13.38,9.55,92
72,"Tunnel Rissani-Merzouga",4.37,15.85,3.98,29.09,24.54,9.03,7.50,37
73,"Tunnel Zagora-M'hamid",0.12,23.78,88.78,30.18,7.06,14.44,1.08,59
74,"Tunnel Ouarzazate-Skoura",0.03,30.27,80.57,20.32,15.71,5.33,6.72,59
75,"Tunnel Kelaa M'gouna-Boumalne",3.27,6.31,91.71,9.03,5.43,0.25,1.58,36
76,"Tunnel Tinghir-Todra",1.54,9.93,97.49,12.28,16.34,7.13,1.34,94
77,"Tunnel Azrou-Ain Leuh",1.40,25.06,47.31,33.10,18.20,10.07,4.63,23
78,"Tunnel Ifrane-Mischliffen",4.75,33.82,42.40,30.44,21.49,10.90,4.18,43
79,"Tunnel Imouzzer-Sefrou",2.78,22.00,80.28,16.41,5.24,5.94,4.55,14
80,"Tunnel FÃ¨s-Saiss",3.13,18.00,65.90,36.79,11.42,2.58,6.61,16
81,"Tunnel MeknÃ¨s-El Hajeb",0.50,15.03,89.38,35.00,6.08,1.01,8.63,91
82,"Tunnel Khemisset-Oulmes",0.50,26.67,70.48,19.24,7.11,0.86,4.07,53
83,"Tunnel Rabat-Temara",1.90,25.75,69.68,8.09,3.73,2.87,6.00,73
84,"Tunnel Harhoura-Ain Aouda",2.17,16.61,66.00,34.38,17.52,5.56,8.27,12
85,"Tunnel Casablanca-Mohammedia",1.36,40.59,1.41,32.44,5.28,11.89,3.21,44
86,"Tunnel Bouskoura-Dar Bouazza",4.37,14.99,91.97,28.64,3.43,0.93,6.13,25
87,"Tunnel Nouaceur-Tit Mellil",1.43,5.22,5.58,5.24,14.14,14.55,6.42,41
88,"Tunnel Mediouna-Benslimane",2.05,30.44,6.42,32.69,8.08,14.03,5.31,20
89,"Tunnel Berrechid-Settat",2.21,2.37,66.12,5.11,18.88,1.78,0.34,7
90,"Tunnel El Gara-Soualem",4.95,47.54,41.45,29.25,4.28,2.98,6.43,11
91,"Tunnel El Jadida-Azemmour",4.21,12.89,18.84,31.63,23.84,12.28,3.72,14
92,"Tunnel Oualidia-Jorf Lasfar",2.09,4.56,25.45,19.18,8.41,10.40,1.95,52
93,"Tunnel Safi-Souira Kedima",4.40,10.23,6.96,13.07,21.28,5.87,1.62,55
94,"Tunnel Chichaoua-Imintanoute",4.52,4.59,72.00,5.95,10.89,11.64,1.75,61
95,"Tunnel Amizmiz-Asni",0.68,38.24,3.30,12.51,5.34,1.80,4.30,94
96,"Tunnel Tahannout-Ourika",1.44,9.96,35.42,38.37,23.45,13.19,5.33,6
97,"Tunnel Essaouira-Tamanar",4.65,22.22,99.23,37.30,2.57,2.11,2.74,93
98,"Tunnel Sidi Bennour-Zemamra",4.63,20.07,33.00,31.76,18.01,10.98,0.73,65
99,"Tunnel Beni Mellal-Azilal",2.74,13.39,60.19,11.82,19.15,14.96,0.90,99
100,"Tunnel Demnate-Imi n'Ifri",1.26,32.44,33.87,32.35,7.22,3.70,0.03,43"""

# Créer le DataFrame
df = pd.read_csv(pd.io.common.StringIO(data))
df_original = df.copy()

# Liste des variables quantitatives
variables = ['Pression terrain (MPa)', 'Convergence (mm)', 'HumiditÃ© (%)', 
             'TempÃ©rature (Â°C)', 'Vibrations (Hz)', 'Fissures (mm)', 
             'Ventilation (m/s)', 'DurÃ©e exploitation (ans)']

# 1. Calcul des moyennes et écarts-types
print("="*80)
print("QUESTION 1 : MOYENNES ET ÉCARTS-TYPES")
print("="*80)

stats_df = pd.DataFrame({
    'Variable': variables,
    'Moyenne': df[variables].mean().values,
    'Écart-type': df[variables].std().values,
    'Min': df[variables].min().values,
    'Max': df[variables].max().values
})

print("\nStatistiques descriptives :")
print(stats_df.to_string(index=False))

print("\n" + "="*80)
print("INTERPRÉTATION DES MOYENNES ET ÉCARTS-TYPES :")
print("="*80)

interpretations = {
    'Pression terrain (MPa)': {
        'moyenne': "La pression moyenne est de 2.31 MPa, ce qui est relativement faible pour un tunnel.",
        'ecart_type': "L'écart-type de 1.47 MPa indique une grande variabilité de pression selon les tunnels."
    },
    'Convergence (mm)': {
        'moyenne': "La convergence moyenne est de 22.09 mm, une valeur modérée.",
        'ecart_type': "L'écart-type élevé (14.27 mm) montre que la déformation varie beaucoup."
    },
    'HumiditÃ© (%)': {
        'moyenne': "Humidité moyenne de 52.35%, typique pour des tunnels souterrains.",
        'ecart_type': "Écart-type de 27.89% indique une grande diversité de conditions d'humidité."
    },
    'TempÃ©rature (Â°C)': {
        'moyenne': "Température moyenne de 21.34°C, confortable pour l'exploitation.",
        'ecart_type': "Écart-type de 9.63°C montre des variations climatiques importantes."
    },
    'Vibrations (Hz)': {
        'moyenne': "Fréquence moyenne des vibrations de 12.77 Hz, acceptable.",
        'ecart_type': "Écart-type de 6.89 Hz indique des sources de vibration très différentes."
    },
    'Fissures (mm)': {
        'moyenne': "Largeur moyenne des fissures de 7.37 mm, nécessite surveillance.",
        'ecart_type': "Écart-type de 4.36 mm montre une hétérogénéité des fissurations."
    },
    'Ventilation (m/s)': {
        'moyenne': "Vitesse moyenne de ventilation de 4.79 m/s, correcte pour l'aération.",
        'ecart_type': "Écart-type de 2.71 m/s indique des systèmes de ventilation différents."
    },
    'DurÃ©e exploitation (ans)': {
        'moyenne': "Durée moyenne d'exploitation de 48.45 ans, certains tunnels sont anciens.",
        'ecart_type': "Écart-type de 30.90 ans montre une grande disparité d'âge des tunnels."
    }
}

for var in variables:
    print(f"\n{var}:")
    print(f"  Moyenne ({stats_df[stats_df['Variable']==var]['Moyenne'].values[0]:.2f}): {interpretations[var]['moyenne']}")
    print(f"  Écart-type ({stats_df[stats_df['Variable']==var]['Écart-type'].values[0]:.2f}): {interpretations[var]['ecart_type']}")

# 2. Matrice centrée réduite (heatmap divisée en 4)
print("\n" + "="*80)
print("QUESTION 2 : MATRICE CENTRÉE RÉDUITE (HEATMAP)")
print("="*80)

# Standardisation des données
scaler = StandardScaler()
data_standardized = scaler.fit_transform(df[variables])
df_standardized = pd.DataFrame(data_standardized, columns=variables, index=df['Nom du Tunnel'])

# Diviser les données en 4 parties (25 tunnels par partie)
n_parts = 4
tunnels_per_part = 25

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Matrice Centrée Réduite des Tunnels (divisée en 4 parties)', fontsize=16, fontweight='bold')

for i in range(n_parts):
    start_idx = i * tunnels_per_part
    end_idx = start_idx + tunnels_per_part
    df_part = df_standardized.iloc[start_idx:end_idx]
    
    ax = axes[i//2, i%2]
    sns.heatmap(df_part, cmap='RdBu_r', center=0, 
                ax=ax, cbar_kws={'label': 'Valeur centrée-réduite'})
    ax.set_title(f'Partie {i+1} : Tunnels {start_idx+1} à {end_idx}')
    ax.set_xlabel('Variables')
    ax.set_ylabel('Tunnels')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()

print("="*80)
print("SUITE DE L'ANALYSE EN COMPOSANTES PRINCIPALES (ACP)")
print("="*80)

# Importations nécessaires
from sklearn.decomposition import PCA
from tabulate import tabulate

# Préparation des données pour l'ACP
X = df[variables].values
noms_tunnels = df['Nom du Tunnel'].values

# 1. Calcul des composantes principales
print("\n" + "="*80)
print("QUESTION 1 : COMPOSANTES PRINCIPALES")
print("="*80)

# Application de l'ACP
pca = PCA(n_components=len(variables))
X_pca = pca.fit_transform(X)

# Création des DataFrames pour les résultats
df_individus = pd.DataFrame({
    'Tunnel': noms_tunnels,
    'CP1': X_pca[:, 0],
    'CP2': X_pca[:, 1],
    'CP3': X_pca[:, 2]
})

# Coordonnées des variables dans l'espace des composantes principales
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

df_variables = pd.DataFrame(
    loadings,
    columns=[f'CP{i+1}' for i in range(len(variables))],
    index=variables
)

print("\nA. COMPOSANTES PRINCIPALES DES INDIVIDUS (premiers 10 tunnels) :")
print("-" * 80)
print(df_individus.head(10).to_string(index=False))

print("\nB. COMPOSANTES PRINCIPALES DES VARIABLES :")
print("-" * 80)
print(df_variables.round(3))

# 2. Plan factoriel des individus
print("\n" + "="*80)
print("QUESTION 2 : PLAN FACTORIEL DES INDIVIDUS")
print("="*80)

# Calcul de la qualité de représentation des individus (cos²)
cos2_individus = X_pca**2 / np.sum(X_pca**2, axis=1, keepdims=True)

# Préparation pour la visualisation
plt.figure(figsize=(14, 10))

# Points colorés par qualité de représentation sur le plan (cos² sur CP1+CP2)
cos2_plan = cos2_individus[:, 0] + cos2_individus[:, 1]
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=cos2_plan, cmap='RdYlGn', 
                     s=80, alpha=0.8, edgecolors='black', linewidth=0.5)

# Ajout des noms des tunnels (uniquement ceux bien représentés)
for i, (x, y) in enumerate(zip(X_pca[:, 0], X_pca[:, 1])):
    cos2_i = cos2_plan[i]
    if cos2_i > 0.6:  # Seuil plus élevé pour éviter la surcharge
        plt.annotate(f"{i+1}", (x, y),  # Affiche juste l'ID
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=7, alpha=0.8, ha='center')

# Cercles de corrélation (cercle unité)
circle = plt.Circle((0, 0), radius=1, 
                   color='blue', fill=False, linestyle='--', alpha=0.3, linewidth=1)
plt.gca().add_patch(circle)

# Axes et grille
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
plt.grid(True, alpha=0.2)

# Labels et titre
plt.xlabel(f'Composante 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'Composante 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.title('Plan Factoriel des Individus (Tunnels)', fontsize=14, fontweight='bold')

# Barre de couleur avec échelle fixe
cbar = plt.colorbar(scatter, boundaries=np.linspace(0, 1, 11))
cbar.set_label('Qualité de représentation (cos² CP1+CP2)', fontsize=10)
cbar.set_ticks(np.linspace(0, 1, 6))
cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

# Échelle ajustée
x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
y_min, y_max = X_pca[:, 1].min(), X_pca[:, 1].max()
margin = 0.1
plt.xlim(x_min - margin, x_max + margin)
plt.ylim(y_min - margin, y_max + margin)

plt.tight_layout()
plt.show()

# 3. Plan factoriel des variables (cercle de corrélation)
print("\n" + "="*80)
print("QUESTION 3 : PLAN FACTORIEL DES VARIABLES (Cercle de Corrélation)")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 10))

# Cercle unité plus visible
circle = plt.Circle((0, 0), radius=1.0, color='navy', 
                   fill=False, linestyle='-', alpha=0.5, linewidth=2)
ax.add_patch(circle)

# Quadrants pour mieux voir les directions
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Variables comme vecteurs avec échelle ajustée
max_loading = np.max(np.abs(df_variables[['CP1', 'CP2']].values))
scale_factor = 1.0 / max_loading  # Pour que les vecteurs atteignent le cercle

for i, var in enumerate(variables):
    x = df_variables.loc[var, 'CP1'] * scale_factor
    y = df_variables.loc[var, 'CP2'] * scale_factor
    
    # Vecteur
    ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, 
            fc='red', ec='red', alpha=0.8, width=0.005)
    
    # Étiquette positionnée intelligemment
    offset_x = 0.05 if x >= 0 else -0.05
    offset_y = 0.05 if y >= 0 else -0.05
    ax.text(x + offset_x, y + offset_y, var, fontsize=10, 
           ha='center', va='center', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Légende des quadrants
ax.text(0.9, 0.02, 'CP1 +', transform=ax.transAxes, fontsize=9, ha='right')
ax.text(0.02, 0.95, 'CP2 +', transform=ax.transAxes, fontsize=9)
ax.text(0.02, 0.02, 'CP1 - / CP2 -', transform=ax.transAxes, fontsize=9)
ax.text(0.9, 0.95, 'CP1 + / CP2 +', transform=ax.transAxes, fontsize=9, ha='right')

# Limites ajustées pour mieux voir
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

# Grille circulaire
for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
    circle_grid = plt.Circle((0, 0), radius=r, color='lightgray', 
                           fill=False, linestyle=':', alpha=0.3)
    ax.add_patch(circle_grid)

ax.grid(True, alpha=0.3)

# Labels et titre
ax.set_xlabel(f'Composante 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'Composante 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('Cercle de Corrélation - Plan Factoriel des Variables', 
            fontsize=14, fontweight='bold')
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

# 4. QUALITÉ DE REPRÉSENTATION EN HEATMAP (divisée en 4 parties)
print("\n" + "="*80)
print("QUESTION 4 : QUALITÉ DE REPRÉSENTATION - HEATMAP (4 parties)")
print("="*80)

# Calcul des cos² pour chaque individu sur chaque axe
cos2_df = pd.DataFrame(cos2_individus[:, :2], 
                      columns=['cos²_Axe1', 'cos²_Axe2'],
                      index=noms_tunnels)
cos2_df['cos²_Plan'] = cos2_df['cos²_Axe1'] + cos2_df['cos²_Axe2']

# Classification de la qualité de représentation avec catégories numériques pour la heatmap
def qualite_categorie(cos2):
    if cos2 >= 0.8:
        return 4  # Très bonne
    elif cos2 >= 0.6:
        return 3  # Bonne
    elif cos2 >= 0.4:
        return 2  # Moyenne
    elif cos2 >= 0.2:
        return 1  # Faible
    else:
        return 0  # Très faible

# Création des catégories pour la heatmap
cos2_cat_df = pd.DataFrame({
    'Axe1_cat': cos2_df['cos²_Axe1'].apply(qualite_categorie),
    'Axe2_cat': cos2_df['cos²_Axe2'].apply(qualite_categorie),
    'Plan_cat': cos2_df['cos²_Plan'].apply(qualite_categorie)
}, index=noms_tunnels)

# HEATMAP DE LA QUALITÉ DE REPRÉSENTATION (divisée en 4 parties)
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Qualité de Représentation des Tunnels - Heatmap (Divisée en 4 parties)', 
             fontsize=16, fontweight='bold')

parties = 4
tunnels_par_partie = 25

# Palette de couleurs pour la qualité
cmap_quality = plt.cm.RdYlGn
norm_quality = plt.Normalize(vmin=0, vmax=4)

# Labels pour la légende
quality_labels = ['Très faible (0-0.2)', 'Faible (0.2-0.4)', 'Moyenne (0.4-0.6)', 
                  'Bonne (0.6-0.8)', 'Très bonne (0.8-1.0)']

for partie in range(parties):
    debut = partie * tunnels_par_partie
    fin = debut + tunnels_par_partie
    
    # Sélection des données pour cette partie
    partie_tunnels = noms_tunnels[debut:fin]
    cos2_partie = cos2_cat_df.loc[partie_tunnels]
    
    # Préparation des données pour la heatmap
    heatmap_data = cos2_partie[['Axe1_cat', 'Axe2_cat', 'Plan_cat']].T
    
    # Création de la heatmap
    ax = axes[partie//2, partie%2]
    im = ax.imshow(heatmap_data, cmap=cmap_quality, aspect='auto', 
                   norm=norm_quality, interpolation='nearest')
    
    # Configuration des axes
    ax.set_title(f'Partie {partie+1} : Tunnels {debut+1} à {fin}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tunnels', fontsize=10)
    ax.set_ylabel('Axe / Plan', fontsize=10)
    
    # Ticks et labels
    ax.set_xticks(range(len(partie_tunnels)))
    ax.set_xticklabels([f'T{i+1}' for i in range(debut, fin)], 
                       rotation=90, fontsize=7)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Axe 1', 'Axe 2', 'Plan CP1+2'], fontsize=10)
    
    # Ajout des valeurs de cos² sur chaque cellule
    cos2_values_partie = cos2_df.loc[partie_tunnels]
    for i in range(3):  # Pour chaque ligne (Axe1, Axe2, Plan)
        for j in range(len(partie_tunnels)):  # Pour chaque tunnel
            if i == 0:
                value = cos2_values_partie.iloc[j, 0]  # cos² Axe1
            elif i == 1:
                value = cos2_values_partie.iloc[j, 1]  # cos² Axe2
            else:
                value = cos2_values_partie.iloc[j, 2]  # cos² Plan
            
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                   fontsize=6, color='black' if value > 0.5 else 'white',
                   fontweight='bold')

# Barre de couleur commune
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
cbar.set_ticklabels(quality_labels)
cbar.set_label('Qualité de Représentation', fontsize=11)

plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()

# 5. TABLEAU DE CONTRIBUTION EN TEXTE (pour référence)
print("\n" + "="*80)
print("TABLEAUX DE CONTRIBUTION (Top 10)")
print("="*80)

# Calcul des contributions des individus
contrib_individus = (X_pca**2) / (len(X) * pca.explained_variance_)
contrib_df = pd.DataFrame(contrib_individus[:, :2],
                         columns=['Contrib_Axe1', 'Contrib_Axe2'],
                         index=noms_tunnels)
contrib_df['Contrib_Plan'] = contrib_df['Contrib_Axe1'] + contrib_df['Contrib_Axe2']

# Contributions des variables
contrib_variables = (pca.components_**2) * pca.explained_variance_[:, np.newaxis]
contrib_vars_df = pd.DataFrame(contrib_variables[:2, :].T,
                              columns=['Contrib_Axe1', 'Contrib_Axe2'],
                              index=variables)
contrib_vars_df['Contrib_Total'] = contrib_vars_df.sum(axis=1)

# A. Top 10 des tunnels les plus contributifs
print("\nA. TOP 10 DES TUNNELS LES PLUS CONTRIBUTIFS :")
print("-" * 80)

# Tri par contribution totale
contrib_individus_trie = contrib_df.sort_values('Contrib_Plan', ascending=False)
top10_contrib = contrib_individus_trie.head(10)

top10_table = []
for i, (tunnel, row) in enumerate(top10_contrib.iterrows(), 1):
    tunnel_id = np.where(noms_tunnels == tunnel)[0][0] + 1
    top10_table.append([
        i,
        tunnel_id,
        tunnel,
        f"{row['Contrib_Axe1']*100:.1f}%",
        f"{row['Contrib_Axe2']*100:.1f}%", 
        f"{row['Contrib_Plan']*100:.1f}%"
    ])

print(tabulate(top10_table, 
              headers=['Rang', 'ID', 'Nom du Tunnel', 'Contrib. Axe1', 'Contrib. Axe2', 'Contrib. Plan'],
              tablefmt='grid'))

# B. Contributions des variables
print("\nB. CONTRIBUTIONS DES VARIABLES AUX AXES :")
print("-" * 80)

# Formatage en pourcentage
contrib_vars_pct = contrib_vars_df.copy()
for col in ['Contrib_Axe1', 'Contrib_Axe2', 'Contrib_Total']:
    contrib_vars_pct[col] = contrib_vars_pct[col] * 100

# Tri par contribution totale
contrib_vars_pct = contrib_vars_pct.sort_values('Contrib_Total', ascending=False)

vars_table = []
for var, row in contrib_vars_pct.iterrows():
    vars_table.append([
        var,
        f"{row['Contrib_Axe1']:.1f}%",
        f"{row['Contrib_Axe2']:.1f}%",
        f"{row['Contrib_Total']:.1f}%"
    ])

print(tabulate(vars_table, 
              headers=['Variable', 'Contrib. Axe1', 'Contrib. Axe2', 'Contrib. Totale'],
              tablefmt='grid'))

# 6. VISUALISATION DES CONTRIBUTIONS
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# A. Contributions des individus (Top 15)
top15_ind = contrib_individus_trie.head(15)
axes[0, 0].barh(range(len(top15_ind)), top15_ind['Contrib_Plan'][::-1] * 100)
axes[0, 0].set_yticks(range(len(top15_ind)))
axes[0, 0].set_yticklabels([f"{np.where(noms_tunnels == name)[0][0]+1}: {name[:15]}..." 
                           for name in top15_ind.index[::-1]])
axes[0, 0].set_xlabel('Contribution au plan factoriel (%)')
axes[0, 0].set_title('Top 15 des tunnels les plus contributifs', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# B. Contributions des variables
axes[0, 1].barh(range(len(contrib_vars_pct)), contrib_vars_pct['Contrib_Total'][::-1])
axes[0, 1].set_yticks(range(len(contrib_vars_pct)))
axes[0, 1].set_yticklabels(contrib_vars_pct.index[::-1])
axes[0, 1].set_xlabel('Contribution totale aux axes principaux (%)')
axes[0, 1].set_title('Contributions des variables', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# C. Distribution des qualités de représentation
qualities = ['Très faible', 'Faible', 'Moyenne', 'Bonne', 'Très bonne']
counts_plan = [
    sum(cos2_df['cos²_Plan'] < 0.2),
    sum((cos2_df['cos²_Plan'] >= 0.2) & (cos2_df['cos²_Plan'] < 0.4)),
    sum((cos2_df['cos²_Plan'] >= 0.4) & (cos2_df['cos²_Plan'] < 0.6)),
    sum((cos2_df['cos²_Plan'] >= 0.6) & (cos2_df['cos²_Plan'] < 0.8)),
    sum(cos2_df['cos²_Plan'] >= 0.8)
]

axes[1, 0].bar(qualities, counts_plan, color=cmap_quality(np.linspace(0, 1, 5)))
axes[1, 0].set_xlabel('Qualité de représentation')
axes[1, 0].set_ylabel('Nombre de tunnels')
axes[1, 0].set_title('Distribution de la qualité sur le plan factoriel', fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)
for i, count in enumerate(counts_plan):
    axes[1, 0].text(i, count + 0.5, str(count), ha='center', fontweight='bold')

# D. Heatmap des corrélations avec les contributions
corr_with_contrib = pd.DataFrame({
    'Variable': variables,
    'Corr_CP1': df_variables['CP1'],
    'Corr_CP2': df_variables['CP2'],
    'Contrib': contrib_vars_pct['Contrib_Total'].values
})

heatmap_data = corr_with_contrib[['Corr_CP1', 'Corr_CP2', 'Contrib']].values.T
im = axes[1, 1].imshow(heatmap_data, cmap='coolwarm', aspect='auto')
axes[1, 1].set_xticks(range(len(variables)))
axes[1, 1].set_xticklabels(variables, rotation=45, ha='right')
axes[1, 1].set_yticks(range(3))
axes[1, 1].set_yticklabels(['Corr. CP1', 'Corr. CP2', 'Contrib.%'])
axes[1, 1].set_title('Corrélations et Contributions', fontweight='bold')

# Ajouter les valeurs
for i in range(3):
    for j in range(len(variables)):
        value = heatmap_data[i, j]
        axes[1, 1].text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color='white' if abs(value) > (0.5 if i<2 else 15) else 'black',
                       fontsize=8)

plt.colorbar(im, ax=axes[1, 1])
plt.tight_layout()
plt.show()

# 7. SYNTHÈSE FINALE
print("\n" + "="*80)
print("SYNTHÈSE FINALE DE L'ANALYSE")
print("="*80)

# Calcul des statistiques résumées
variance_explained = pca.explained_variance_ratio_[:2] * 100
cumulative_variance = sum(variance_explained)

print(f"\n1. VARIANCE EXPLIQUÉE :")
print(f"   • Composante 1 : {variance_explained[0]:.1f}%")
print(f"   • Composante 2 : {variance_explained[1]:.1f}%")
print(f"   • Total plan CP1+CP2 : {cumulative_variance:.1f}%")

print(f"\n2. QUALITÉ DE REPRÉSENTATION MOYENNE :")
print(f"   • Axe 1 : {cos2_df['cos²_Axe1'].mean():.3f}")
print(f"   • Axe 2 : {cos2_df['cos²_Axe2'].mean():.3f}")
print(f"   • Plan : {cos2_df['cos²_Plan'].mean():.3f}")

print(f"\n3. TUNNELS BIEN REPRÉSENTÉS (cos² ≥ 0.6) :")
print(f"   • Sur l'axe 1 : {sum(cos2_df['cos²_Axe1'] >= 0.6)} tunnels")
print(f"   • Sur l'axe 2 : {sum(cos2_df['cos²_Axe2'] >= 0.6)} tunnels")
print(f"   • Sur le plan : {sum(cos2_df['cos²_Plan'] >= 0.6)} tunnels")

print(f"\n4. VARIABLES LES PLUS IMPORTANTES :")
top_vars = contrib_vars_pct.head(3)
for i, (var, row) in enumerate(top_vars.iterrows(), 1):
    print(f"   {i}. {var} : {row['Contrib_Total']:.1f}% de contribution")

print(f"\n5. CONCLUSION :")
print(f"   Le plan factoriel explique {cumulative_variance:.1f}% de la variance totale.")
print(f"   {sum(cos2_df['cos²_Plan'] >= 0.6)} tunnels sur 100 sont bien représentés.")
print(f"   L'analyse permet d'identifier les tunnels caractéristiques et les variables discriminantes.")

print("\n" + "="*80)
print("ANALYSE ACP TERMINÉE AVEC SUCCÈS !")
print("="*80)