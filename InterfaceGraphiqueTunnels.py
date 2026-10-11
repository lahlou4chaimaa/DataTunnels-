import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import sys
import os

# Imports pour l'analyse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score)
from scipy.stats import chi2_contingency
from PIL import Image, ImageTk
import warnings
warnings.filterwarnings('ignore')

class ModernButton(tk.Canvas):
    """Bouton moderne avec effets hover et animations"""
    def __init__(self, parent, text, command, color="#2563eb", hover_color="#1d4ed8", **kwargs):
        super().__init__(parent, highlightthickness=0, **kwargs)
        self.color = color
        self.hover_color = hover_color
        self.command = command
        self.text = text
        
        self.draw_button(self.color)
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        
    def draw_button(self, color):
        self.delete("all")
        width = int(self.cget("width"))
        height = int(self.cget("height"))
        
        self.create_rounded_rect(5, 5, width-5, height-5, 15, fill=color, outline="")
        self.create_text(width//2, height//2, text=self.text, 
                        fill="white", font=("Segoe UI", 12, "bold"))
    
    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1,
                 x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2, y2,
                 x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2, x1, y2,
                 x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def on_enter(self, e):
        self.draw_button(self.hover_color)
        self.config(cursor="hand2")
    
    def on_leave(self, e):
        self.draw_button(self.color)
        self.config(cursor="")
    
    def on_click(self, e):
        if self.command:
            self.command()

class DataManager:
    """Gestionnaire de données centralisé avec toutes les analyses"""
    def __init__(self):
        self.df = None
        self.variables = None
        self.X = None
        self.X_scaled = None
        self.scaler = None
        self.pca = None
        self.pca_results = {}
        self.kmeans_models = {}
        self.rf_models = {}
        self.cyber_results = {}
        self.afc_results = {}
        
    def load_data(self, file_path):
        """Charge les données depuis un fichier CSV"""
        try:
            self.df = pd.read_csv(file_path)
            self.variables = ['Pression terrain (MPa)', 'Convergence (mm)', 'Humidité (%)', 
                             'Température (°C)', 'Vibrations (Hz)', 'Fissures (mm)', 
                             'Ventilation (m/s)', 'Durée exploitation (ans)']
            self.X = self.df[self.variables]
            return True
        except Exception as e:
            print(f"Erreur chargement données: {e}")
            return False
    
    def run_acp_full(self):
        """Exécute l'ACP complète comme dans le code original"""
        try:
            # Standardisation
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X)
            
            # ACP
            self.pca = PCA()
            X_pca = self.pca.fit_transform(self.X_scaled)
            
            # Stocker les résultats
            self.pca_results['valeurs_propres'] = self.pca.explained_variance_
            self.pca_results['variance_expliquee'] = self.pca.explained_variance_ratio_ * 100
            self.pca_results['variance_cumulee'] = np.cumsum(self.pca_results['variance_expliquee'])
            self.pca_results['X_pca'] = X_pca
            self.pca_results['n_composantes_kaiser'] = np.sum(self.pca_results['valeurs_propres'] >= 1)
            self.pca_results['corr_matrix'] = self.X.corr()
            
            # Coordonnées des individus
            n_kaiser = self.pca_results['n_composantes_kaiser']
            self.pca_results['coords_individus'] = pd.DataFrame(
                X_pca[:, :n_kaiser],
                columns=[f'PC{i+1}' for i in range(n_kaiser)],
                index=self.df['Nom du Tunnel']
            )
            
            # Calcul des cos²
            cos2 = (X_pca**2) / np.sum(self.X_scaled**2, axis=1)[:, np.newaxis]
            self.pca_results['cos2'] = cos2
            
            # Corrélations des variables
            correlations = self.pca.components_.T * np.sqrt(self.pca_results['valeurs_propres'])
            self.pca_results['correlations'] = correlations
            
            return True
        except Exception as e:
            print(f"Erreur ACP: {e}")
            return False
    
    def run_kmeans(self, n_clusters):
        """Exécute K-Means comme dans le code original"""
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(self.X_scaled)
            
            # Centres des clusters en valeurs originales
            centres_originaux = self.scaler.inverse_transform(kmeans.cluster_centers_)
            
            # Métriques
            silhouette = silhouette_score(self.X_scaled, clusters)
            davies = davies_bouldin_score(self.X_scaled, clusters)
            calinski = calinski_harabasz_score(self.X_scaled, clusters)
            
            self.kmeans_models[f'K{n_clusters}'] = {
                'model': kmeans,
                'clusters': clusters,
                'centres_originaux': centres_originaux,
                'silhouette': silhouette,
                'davies': davies,
                'calinski': calinski
            }
            
            return True
        except Exception as e:
            print(f"Erreur K-Means: {e}")
            return False
    
    def run_random_forest(self, data_type='original'):
        """Exécute Random Forest comme dans le code original"""
        try:
            # Préparation des données selon le type
            if data_type == 'original':
                X_data = self.X.drop('Durée exploitation (ans)', axis=1)
                y_data = self.X['Durée exploitation (ans)']
            elif data_type == 'scaled':
                X_scaled_df = pd.DataFrame(self.X_scaled, columns=self.variables)
                X_data = X_scaled_df.drop('Durée exploitation (ans)', axis=1)
                y_data = self.X['Durée exploitation (ans)']
            else:  # pca
                n_comp = self.pca_results['n_composantes_kaiser']
                X_data = pd.DataFrame(
                    self.pca_results['X_pca'][:, :n_comp],
                    columns=[f'PC{i+1}' for i in range(n_comp)]
                )
                y_data = self.X['Durée exploitation (ans)']
            
            # Split des données
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, test_size=0.30, random_state=42
            )
            
            # Entraînement du modèle
            rf = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            rf.fit(X_train, y_train)
            
            # Prédictions et métriques
            y_train_pred = rf.predict(X_train)
            y_test_pred = rf.predict(X_test)
            
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            self.rf_models[data_type] = {
                'model': rf,
                'X_test': X_test,
                'y_test': y_test,
                'y_test_pred': y_test_pred,
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'feature_importance': pd.DataFrame({
                    'Feature': X_data.columns,
                    'Importance': rf.feature_importances_
                }).sort_values('Importance', ascending=False)
            }
            
            return True
        except Exception as e:
            print(f"Erreur Random Forest: {e}")
            return False
    
    def run_cyber_analysis(self):
        """Exécute l'analyse de cybersécurité comme dans le code original"""
        try:
            # Sélection des variables
            features_cyber = self.variables
            
            # Normalisation Min-Max
            scaler_cyber = MinMaxScaler()
            X_scaled_cyber = scaler_cyber.fit_transform(self.X[features_cyber])
            
            # Isolation Forest
            iso_cyber = IsolationForest(contamination=0.15, random_state=42)
            iso_pred = iso_cyber.fit_predict(X_scaled_cyber)
            
            # Local Outlier Factor
            lof_cyber = LocalOutlierFactor(n_neighbors=5, contamination=0.15)
            lof_pred = lof_cyber.fit_predict(X_scaled_cyber)
            
            self.cyber_results = {
                'iso_pred': iso_pred,
                'lof_pred': lof_pred,
                'X_scaled_cyber': X_scaled_cyber,
                'features_cyber': features_cyber
            }
            
            return True
        except Exception as e:
            print(f"Erreur analyse cyber: {e}")
            return False
    
    def run_afc_analysis(self):
        """Exécute l'analyse AFC comme dans le code original"""
        try:
            # Discrétisation des 8 premières variables
            discret_df = self.X.iloc[:, :8].copy()
            
            def discretize_var(col):
                return pd.cut(col, bins=3, labels=['B','M','H'])
            
            discrete_df = discret_df.apply(discretize_var)
            discrete_df.columns = [f'Var{i+1}' for i in range(8)]
            
            # Création tableau disjonctif
            individus = discrete_df.index
            modalites = []
            for col in discrete_df.columns:
                modalites.extend([f"{col}_B", f"{col}_M", f"{col}_H"])
            
            tableau_disjonctif = pd.DataFrame(0, index=individus, columns=modalites)
            
            # Remplissage
            for idx, row in discrete_df.iterrows():
                for col in discrete_df.columns:
                    modalite = f"{col}_{row[col]}"
                    tableau_disjonctif.loc[idx, modalite] = 1
            
            # Calculs AFC
            F = tableau_disjonctif.values.astype(float)
            n = F.sum()
            masses_lignes = F.sum(axis=1) / n
            masses_colonnes = F.sum(axis=0) / n
            E = np.outer(masses_lignes, masses_colonnes) * n
            ecarts = F - E
            
            # SVD
            U, s, Vt = np.linalg.svd(ecarts, full_matrices=False)
            valeurs_propres = s**2
            inertie_expl = 100 * valeurs_propres / valeurs_propres.sum()
            
            # Coordonnées
            coords_individus = pd.DataFrame(
                U[:, :2] * s[:2], 
                index=tableau_disjonctif.index, 
                columns=['F1', 'F2']
            )
            
            coords_modalites = pd.DataFrame(
                Vt[:2, :].T * s[:2], 
                index=tableau_disjonctif.columns, 
                columns=['F1', 'F2']
            )
            
            self.afc_results = {
                'discrete_df': discrete_df,
                'tableau_disjonctif': tableau_disjonctif,
                'coords_individus': coords_individus,
                'coords_modalites': coords_modalites,
                'valeurs_propres': valeurs_propres,
                'inertie_expl': inertie_expl
            }
            
            return True
        except Exception as e:
            print(f"Erreur AFC: {e}")
            return False

class TunnelAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analyse Complète des Tunnels Marocains")
        
        # Configuration fenêtre
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 1400
        window_height = 900
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(True, True)
        
        # Couleurs modernes
        self.colors = {
            'primary': '#1e40af',      # Bleu foncé
            'secondary': '#3b82f6',    # Bleu
            'accent': '#10b981',       # Vert émeraude
            'warning': '#f59e0b',      # Orange
            'danger': '#ef4444',       # Rouge
            'dark': '#111827',         # Gris très foncé
            'light': '#f8fafc',        # Blanc bleuté
            'white': '#ffffff'
        }
        
        self.root.configure(bg=self.colors['dark'])
        
        # Gestionnaire de données
        self.data_manager = DataManager()
        self.current_page = None
        self.current_analysis = None
        
        # Créer page d'accueil
        self.create_welcome_page()
    
    def clear_window(self):
        """Efface tous les widgets"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def create_welcome_page(self):
        """Page d'accueil moderne"""
        self.clear_window()
        self.current_page = "welcome"
        
        # Fond principal
        main_frame = tk.Frame(self.root, bg=self.colors['dark'])
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Conteneur central
        center_container = tk.Frame(main_frame, bg=self.colors['dark'])
        center_container.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Logo/Icone
        icon_label = tk.Label(center_container,
                            text="🔬",
                            font=("Segoe UI", 72),
                            bg=self.colors['dark'],
                            fg=self.colors['accent'])
        icon_label.pack(pady=(0, 30))
        
        # Titre principal
        title_label = tk.Label(center_container,
                              text="Analysis of Moroccan Tunnels",
                              font=("Segoe UI", 36, "bold"),
                              bg=self.colors['dark'],
                              fg=self.colors['white'],
                              justify=tk.CENTER)
        title_label.pack(pady=(0, 20))
        
        # Sous-titre
        subtitle_label = tk.Label(center_container,
                                 text="Data Science • Intelligence Artificielle • Cybersécurité",
                                 font=("Segoe UI", 16),
                                 bg=self.colors['dark'],
                                 fg=self.colors['light'])
        subtitle_label.pack(pady=(0, 60))
        
        # Informations
        info_frame = tk.Frame(center_container, bg=self.colors['dark'])
        info_frame.pack(pady=(0, 60))
        
        professor_label = tk.Label(info_frame,
                                  text="Professor: Dr. EL MKHALET MOUNA",
                                  font=("Segoe UI", 14),
                                  bg=self.colors['dark'],
                                  fg="#9ca3af")
        professor_label.pack()
        
        student_label = tk.Label(info_frame,
                                text="Student: Lahlou Chaimaa",
                                font=("Segoe UI", 14),
                                bg=self.colors['dark'],
                                fg="#9ca3af")
        student_label.pack(pady=(10, 0))
        
        # Bouton START
        start_btn = ModernButton(center_container,
                                "🚀 Start ",
                                self.create_main_dashboard,
                                color=self.colors['accent'],
                                hover_color="#059669",
                                width=350, height=60)
        start_btn.pack()
        
        # Footer
        footer_frame = tk.Frame(main_frame, bg='#030712', height=50)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        footer_frame.pack_propagate(False)
        
        footer_label = tk.Label(footer_frame,
                               text="© 2025 - Projet Final Data Analyst",
                               font=("Segoe UI", 10),
                               bg='#030712',
                               fg='#6b7280')
        footer_label.pack(pady=15)
    
    def create_main_dashboard(self):
        """Tableau de bord principal"""
        self.clear_window()
        self.current_page = "dashboard"
        
        # Conteneur principal
        main_frame = tk.Frame(self.root, bg=self.colors['dark'])
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['primary'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Bouton retour
        back_btn = ModernButton(header_frame, "🏠", 
                               self.create_welcome_page,
                               color="#374151", hover_color="#1f2937",
                               width=60, height=40)
        back_btn.place(x=20, y=20)
        
        # Titre
        title_label = tk.Label(header_frame,
                              text="Dashboard – Tunnel Analysis",
                              font=("Segoe UI", 20, "bold"),
                              bg=self.colors['primary'],
                              fg='white')
        title_label.pack(pady=25)
        
        # Bouton charger données
        load_btn = ModernButton(header_frame, "📂 Upload Data", 
                               self.load_data_file,
                               color=self.colors['accent'],
                               hover_color="#059669",
                               width=200, height=45)
        load_btn.place(relx=1.0, x=-20, y=18, anchor=tk.NE)
        
        # Contenu principal
        content_frame = tk.Frame(main_frame, bg=self.colors['dark'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Sections d'analyse
        sections = [
            {
                'title': '📊 Data-Analyst ACP ',
                'description': 'Analyse en Composantes Principales (8 analyses)',
                'color': self.colors['secondary'],
                'command': self.open_acp_dashboard
            },
            {
                'title': '🤖 Artifical Intelligence Clustering and Forecasting  ',
                'description': 'K-Means & Random Forest (6 analyses)',
                'color': self.colors['accent'],
                'command': self.open_ai_dashboard
            },
            {
                'title': '📈 Data-Analyst AFC',
                'description': 'Analyse Factorielle des Correspondances (3 analyses)',
                'color': self.colors['warning'],
                'command': self.open_afc_dashboard
            },
            {
                'title': '🔒 Cyber-Security',
                'description': 'Détection d\'Anomalies (3 analyses)',
                'color': self.colors['danger'],
                'command': self.open_cyber_dashboard
            }
        ]
        
        # Création des cartes
        for i, section in enumerate(sections):
            row = i // 2
            col = i % 2
            
            card_frame = tk.Frame(content_frame, bg=self.colors['dark'])
            card_frame.grid(row=row, column=col, padx=20, pady=20, sticky="nsew")
            
            card = tk.Frame(card_frame, bg='#1f2937', relief=tk.RAISED, bd=0)
            card.pack(fill=tk.BOTH, expand=True)
            
            # Titre de la section
            title_label = tk.Label(card,
                                  text=section['title'],
                                  font=("Segoe UI", 16, "bold"),
                                  bg='#1f2937',
                                  fg=section['color'],
                                  justify=tk.LEFT)
            title_label.pack(anchor='w', padx=30, pady=(30, 10))
            
            # Description
            desc_label = tk.Label(card,
                                 text=section['description'],
                                 font=("Segoe UI", 11),
                                 bg='#1f2937',
                                 fg='#9ca3af',
                                 wraplength=250,
                                 justify=tk.LEFT)
            desc_label.pack(anchor='w', padx=30, pady=(0, 30))
            
            # Bouton d'accès
            access_btn = ModernButton(card,
                                     "ACCÉDER →",
                                     section['command'],
                                     color=section['color'],
                                     hover_color=self.darken_color(section['color'], 20),
                                     width=200, height=45)
            access_btn.pack(pady=(0, 30))
        
        # Configuration de la grille
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_rowconfigure(1, weight=1)
    
    def open_acp_dashboard(self):
        """Dashboard ACP - MODIFIÉ: Boutons plus grands et centrés"""
        analyses = [
            {'title': '1. Statistiques Descriptives', 'description': 'Moyennes, écarts-types et CV', 'command': self.show_acp_stats},
            {'title': '2. Matrice Centrée Réduite', 'description': '4 heatmaps des tunnels', 'command': self.show_matrice_centree},
            {'title': '3. Matrice de Corrélation', 'description': 'Corrélations entre variables', 'command': self.show_correlation_matrix},
            {'title': '4. Valeurs Propres & Inerties', 'description': 'Scree plot et variance', 'command': self.show_valeurs_propres},
            {'title': '5. Qualité de Représentation', 'description': 'Cos² pour chaque tunnel', 'command': self.show_qualite_representation},
            {'title': '6. Coordonnées Individus', 'description': 'Coordonnées sur composantes', 'command': self.show_coordonnees_individus},
            {'title': '7. Plan Factoriel Individus', 'description': 'Projection PC1-PC2 avec noms', 'command': self.show_plan_individus},
            {'title': '8. Cercle de Corrélation', 'description': 'Projection des variables', 'command': self.show_cercle_correlation}
        ]
        self.create_module_dashboard_acp("📊 ANALYSE ACP COMPLÈTE", analyses, self.colors['secondary'], "acp")
    
    def open_ai_dashboard(self):
        """Dashboard IA"""
        analyses = [
            {'title': '1. K-Means K=3', 'description': 'Clustering en 3 groupes', 'command': lambda: self.show_kmeans(3)},
            {'title': '2. K-Means K=4', 'description': 'Clustering en 4 groupes', 'command': lambda: self.show_kmeans(4)},
            {'title': '3. Comparaison K=3 vs K=4', 'description': 'Analyse comparative', 'command': self.show_kmeans_comparison},
            {'title': '4. Random Forest Original', 'description': 'Prédiction données brutes', 'command': lambda: self.show_random_forest('original')},
            {'title': '5. Random Forest Standardisé', 'description': 'Prédiction données standardisées', 'command': lambda: self.show_random_forest('scaled')},
            {'title': '6. Random Forest ACP', 'description': 'Prédiction composantes principales', 'command': lambda: self.show_random_forest('pca')}
        ]
        self.create_module_dashboard("🤖 Artificial Intelligence", analyses, self.colors['accent'], "ai")
    
    def open_afc_dashboard(self):
        """Dashboard AFC"""
        analyses = [
            {'title': '1. Discrétisation', 'description': 'Variables en modalités B/M/H', 'command': self.show_afc_discretisation},
            {'title': '2. Tableau Disjonctif', 'description': 'Matrice 100×24', 'command': self.show_afc_tableau},
            {'title': '3. Plan Factoriel AFC', 'description': 'Visualisation F1×F2', 'command': self.show_afc_plan}
        ]
        self.create_module_dashboard("📈 ANALYSE AFC", analyses, self.colors['warning'], "afc")
    
    def open_cyber_dashboard(self):
        """Dashboard Cybersécurité"""
        analyses = [
            {'title': '1. Isolation Forest', 'description': 'Détection anomalies IF', 'command': self.show_isolation_forest},
            {'title': '2. Local Outlier Factor', 'description': 'Détection anomalies LOF', 'command': self.show_lof},
            {'title': '3. Comparaison IF vs LOF', 'description': 'Analyse consensus', 'command': self.show_cyber_comparison}
        ]
        self.create_module_dashboard("🔒 CYBERSÉCURITÉ", analyses, self.colors['danger'], "cyber")
    
    def create_module_dashboard_acp(self, title, analyses, color, module_name):
        """Dashboard de module ACP MODIFIÉ avec boutons plus grands et centrés"""
        self.clear_window()
        self.current_page = module_name
        
        # Conteneur principal
        main_frame = tk.Frame(self.root, bg=self.colors['dark'])
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=color, height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Bouton retour
        back_btn = ModernButton(header_frame, "← RETOUR", 
                               self.create_main_dashboard,
                               color="#374151", hover_color="#1f2937",
                               width=150, height=40)
        back_btn.place(x=20, y=20)
        
        # Titre
        title_label = tk.Label(header_frame,
                              text=title,
                              font=("Segoe UI", 20, "bold"),
                              bg=color,
                              fg='white')
        title_label.pack(pady=25)
        
        # Conteneur pour les boutons avec scrollbar
        canvas = tk.Canvas(main_frame, bg=self.colors['dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['dark'])
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Conteneur pour centrer les boutons
        center_container = tk.Frame(scrollable_frame, bg=self.colors['dark'])
        center_container.pack(expand=True, pady=40)
        
        # Calculer le nombre de colonnes (4 colonnes pour 8 boutons)
        n_cols = 2
        
        # Organiser les 8 boutons en 4 colonnes (2x4 grid)
        for i, analysis in enumerate(analyses):
            row = i // n_cols
            col = i % n_cols
            
            # Créer un cadre pour chaque bouton avec une taille fixe
            btn_container = tk.Frame(center_container, bg=self.colors['dark'])
            btn_container.grid(row=row, column=col, padx=50, pady=30, sticky="nsew")
            
            # Créer le bouton avec une taille plus grande (450x80 au lieu de 400x60)
            btn = ModernButton(btn_container,
                              analysis['title'],
                              analysis['command'],
                              color=color,
                              hover_color=self.darken_color(color, 20),
                              width=450, height=80)  # Taille augmentée
            btn.pack(fill=tk.BOTH, expand=True)
            
            # Description
            desc_label = tk.Label(btn_container,
                                 text=analysis['description'],
                                 font=("Segoe UI", 10),
                                 bg=self.colors['dark'],
                                 fg='#9ca3af',
                                 justify=tk.CENTER)
            desc_label.pack(pady=(10, 0))
        
        # Configurer les poids des colonnes pour centrer
        for i in range(n_cols):
            center_container.grid_columnconfigure(i, weight=1)
        
        # Configurer les poids des lignes
        for i in range((len(analyses) + n_cols - 1) // n_cols):
            center_container.grid_rowconfigure(i, weight=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_module_dashboard(self, title, analyses, color, module_name):
        """Dashboard de module avec boutons verticaux (pour les autres modules)"""
        self.clear_window()
        self.current_page = module_name
        
        # Conteneur principal
        main_frame = tk.Frame(self.root, bg=self.colors['dark'])
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=color, height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Bouton retour
        back_btn = ModernButton(header_frame, "← RETOUR", 
                               self.create_main_dashboard,
                               color="#374151", hover_color="#1f2937",
                               width=150, height=40)
        back_btn.place(x=20, y=20)
        
        # Titre
        title_label = tk.Label(header_frame,
                              text=title,
                              font=("Segoe UI", 20, "bold"),
                              bg=color,
                              fg='white')
        title_label.pack(pady=25)
        
        # Contenu avec scrollbar
        canvas = tk.Canvas(main_frame, bg=self.colors['dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['dark'])
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Organiser les boutons en 2 colonnes
        num_analyses = len(analyses)
        half = (num_analyses + 1) // 2
        
        for i in range(num_analyses):
            if i < half:
                column = 0
                row = i
            else:
                column = 1
                row = i - half
            
            analysis = analyses[i]
            
            # Créer le bouton d'analyse
            btn_frame = tk.Frame(scrollable_frame, bg=self.colors['dark'])
            btn_frame.grid(row=row, column=column, padx=20, pady=15, sticky="nsew")
            
            btn = ModernButton(btn_frame,
                              analysis['title'],
                              analysis['command'],
                              color=color,
                              hover_color=self.darken_color(color, 20),
                              width=400, height=60)
            btn.pack(fill=tk.X)
            
            # Description
            desc_label = tk.Label(btn_frame,
                                 text=analysis['description'],
                                 font=("Segoe UI", 9),
                                 bg=self.colors['dark'],
                                 fg='#9ca3af',
                                 justify=tk.LEFT)
            desc_label.pack(anchor='w', padx=10, pady=(5, 0))
        
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    # ==================== MÉTHODES D'ANALYSE ====================
    
    def show_acp_stats(self):
        """Statistiques descriptives"""
        self.show_loading("Calcul des statistiques...")
        
        def compute():
            try:
                if self.data_manager.df is None:
                    self.root.after(0, lambda: messagebox.showerror("Erreur", "Données non chargées"))
                    return
                
               # Calcul avec index cohérent (colonnes de X)
                variables = X.columns.tolist()  # Utiliser les vraies colonnes
                moyennes = X.mean()
                ecarts_types = X.std()
            
            # CV avec protection division par zéro
                cv = pd.Series(index=variables, dtype=float)
                for var in variables:
                    if moyennes[var] == 0:
                        cv[var] = float('inf') if ecarts_types[var] > 0 else 0
                    else:
                        cv[var] = (ecarts_types[var] / moyennes[var]) * 100
                
                stats_df = pd.DataFrame({
                    'Variable': self.data_manager.variables,
                    'Moyenne': moyennes.values,
                    'Écart-Type': ecarts_types.values,
                    'CV (%)': cv.values
                })
                
                interpret_text = "📝 INTERPRÉTATIONS DES COEFFICIENTS DE VARIATION:\n\n"
                for var in self.data_manager.variables:
                    moy = moyennes[var]
                    std = ecarts_types[var]
                    cv_val = std / moy * 100
                    
                    interpret_text += f"• {var}:\n"
                    interpret_text += f"  - Moyenne: {moy:.2f}\n"
                    interpret_text += f"  - Écart-Type: {std:.2f}\n"
                    interpret_text += f"  - Coefficient de variation: {cv_val:.2f}%\n"
                    
                    if cv_val < 15:
                        interpret_text += f"  → Variabilité FAIBLE: Les tunnels sont homogènes\n\n"
                    elif cv_val < 30:
                        interpret_text += f"  → Variabilité MODÉRÉE: Dispersion moyenne\n\n"
                    else:
                        interpret_text += f"  → Variabilité FORTE: Grande hétérogénéité\n\n"
                
                self.root.after(0, lambda: self.display_dataframe_result(
                    "STATISTIQUES DESCRIPTIVES", stats_df, interpret_text))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()


    
    def show_matrice_centree(self):
        """Matrice centrée réduite"""
        self.show_loading("Calcul de la matrice centrée réduite...")
        
        def compute():
            try:
                if self.data_manager.df is None:
                    self.root.after(0, lambda: messagebox.showerror("Erreur", "Données non chargées"))
                    return
                
                if self.data_manager.X_scaled is None:
                    self.data_manager.run_acp_full()
                
                matrice_cr = pd.DataFrame(
                    self.data_manager.X_scaled,
                    columns=self.data_manager.variables,
                    index=self.data_manager.df['Nom du Tunnel']
                )
                
                fig, axes = plt.subplots(2, 2, figsize=(25, 25))
                fig.suptitle('MATRICE CENTRÉE RÉDUITE - 100 TUNNELS × 8 VARIABLES', 
                            fontsize=10, fontweight='bold', y=0.99)
                
                # Heatmap 1: Tunnels 1-25
                sns.heatmap(matrice_cr.iloc[:25].round(2), annot=True, fmt='.2f', 
                           cmap='RdBu_r', center=0, ax=axes[0,0], cbar=True, 
                           annot_kws={'size': 5}, linewidths=1, linecolor='white')

                
                # Heatmap 2: Tunnels 26-50
                sns.heatmap(matrice_cr.iloc[25:50].round(2), annot=True, fmt='.2f', 
                           cmap='RdBu_r', center=0, ax=axes[0,1], cbar=True,
                           annot_kws={'size': 5}, linewidths=1, linecolor='white')

                
                # Heatmap 3: Tunnels 51-75
                sns.heatmap(matrice_cr.iloc[50:75].round(2), annot=True, fmt='.2f', 
                           cmap='RdBu_r', center=0, ax=axes[1,0], cbar=True,
                           annot_kws={'size': 5}, linewidths=1, linecolor='white')
               
                
                # Heatmap 4: Tunnels 76-100
                sns.heatmap(matrice_cr.iloc[75:].round(2), annot=True, fmt='.2f', 
                           cmap='RdBu_r', center=0, ax=axes[1,1], cbar=True,
                           annot_kws={'size': 5}, linewidths=1, linecolor='white')

                
                plt.tight_layout(rect=[0, 0, 1, 0.99])
                
                info_text = "✅ Matrice centrée réduite calculée (100 tunnels × 8 variables)\n\n"
                info_text += "📊 4 HEATMAPS GÉNÉRÉES - TOUS LES 100 TUNNELS VISUALISÉS!\n"
                info_text += "💡 Valeurs lisibles avec espacement optimal\n"
                
                self.root.after(0, lambda: self.display_matplotlib_result(
                    "MATRICE CENTRÉE RÉDUITE", fig, info_text))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_correlation_matrix(self):
     """Matrice de corrélation - VERSION ROBUSTE"""
     self.show_loading("Calcul de la matrice de corrélation...")
     
     def compute():
         try:
             if self.data_manager.df is None:
                 self.root.after(0, lambda: messagebox.showerror("Erreur", "Données non chargées"))
                 return
             
             # VÉRIFICATION DES DONNÉES
             if self.data_manager.X is None or self.data_manager.X.empty:
                 self.root.after(0, lambda: messagebox.showerror("Erreur", "self.data_manager.X est vide"))
                 return
             
             # NETTOYAGE AUTOMATIQUE
             X_clean = self.data_manager.X.select_dtypes(include=[np.number])
             X_clean = X_clean.dropna(axis=1, how='all')
             X_clean = X_clean.fillna(X_clean.mean())
             
             if X_clean.empty:
                 self.root.after(0, lambda: messagebox.showerror("Erreur", "Aucune donnée numérique"))
                 return
             
             # MATRICE DE CORRÉLATION
             corr_matrix = X_clean.corr()
             
             # FIGURE
             fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20), 
                                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})
             ax2.axis('off')
             
             # HEATMAP
             labels = [col.split('(')[0].strip() for col in X_clean.columns]
             sns.heatmap(corr_matrix.round(3), annot=True, fmt='.3f', cmap='coolwarm',
                        center=0, square=True, ax=ax1, cbar_kws={"shrink": 0.8},
                        annot_kws={'size': 14})
             
             ax1.set_title('MATRICE DE CORRÉLATION - Tunnels Marocains', 
                          fontsize=16, fontweight='bold', pad=20)
             ax1.tick_params(axis='x', rotation=45)
             
             # ANALYSE TEXTE
             top_corrs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
             top_corrs = top_corrs[top_corrs < 1].head(10)
             
             
             plt.tight_layout()
             self.root.after(0, lambda: self.display_matplotlib_result(
                 "MATRICE DE CORRÉLATION", fig, f"✅ {len(X_clean.columns)} variables"))
                 
         except Exception as e:
             self.root.after(0, lambda: messagebox.showerror("Erreur", f"{str(e)}"))
     
     threading.Thread(target=compute, daemon=True).start()


    
    def display_correlation_analysis(self, title, fig, analysis_text, corr_matrix):
        """Affiche l'analyse de corrélation dans une fenêtre dédiée"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg=self.colors['dark'])
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['primary'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        back_btn = ModernButton(header_frame, "← RETOUR", 
                               lambda: self.return_to_module(self.current_page),
                               color="#374151", hover_color="#1f2937",
                               width=150, height=40)
        back_btn.place(x=20, y=20)
        
        tk.Label(header_frame, text=title,
                font=("Segoe UI", 18, "bold"),
                bg=self.colors['primary'], fg='white').pack(pady=25)
        
        # Contenu avec notebook pour séparer graphique et texte
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Onglet 1: Graphique
        tab1_frame = tk.Frame(notebook, bg=self.colors['dark'])
        notebook.add(tab1_frame, text="📊 Matrice de Corrélation")
        
        canvas = FigureCanvasTkAgg(fig, master=tab1_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Onglet 2: Analyse textuelle
        tab2_frame = tk.Frame(notebook, bg=self.colors['dark'])
        notebook.add(tab2_frame, text="📝 Analyse Détailée")
        
        text_widget = scrolledtext.ScrolledText(tab2_frame,
                                              font=("Segoe UI", 10),
                                              bg=self.colors['dark'],
                                              fg='white')
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert('1.0', analysis_text)
        text_widget.config(state='disabled')
        
        # Onglet 3: Données brutes
        tab3_frame = tk.Frame(notebook, bg=self.colors['dark'])
        notebook.add(tab3_frame, text="📋 Données Brutes")
        
        # Créer Treeview pour afficher la matrice
        tree_frame = tk.Frame(tab3_frame, bg=self.colors['dark'])
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        tree = ttk.Treeview(tree_frame, 
                           yscrollcommand=vsb.set, 
                           xscrollcommand=hsb.set,
                           style="Custom.Treeview")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Treeview",
                       background='#1f2937',
                       foreground='white',
                       fieldbackground='#1f2937',
                       rowheight=25)
        style.configure("Custom.Treeview.Heading",
                       background=self.colors['primary'],
                       foreground='white',
                       relief='flat')
        
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)
        
        tree["columns"] = list(corr_matrix.columns)
        tree["show"] = "headings"
        
        for col in corr_matrix.columns:
            short_name = col.split('(')[0].strip()
            tree.heading(col, text=short_name)
            tree.column(col, width=120, anchor='center')
        
        for idx in corr_matrix.index:
            values = [f"{corr_matrix.loc[idx, col]:.3f}" for col in corr_matrix.columns]
            tree.insert("", "end", values=values)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Boutons
        btn_frame = tk.Frame(main_frame, bg=self.colors['dark'])
        btn_frame.pack(fill=tk.X, pady=10)
        
        save_btn = ModernButton(btn_frame, "💾 SAUVEGARDER",
                               lambda: self.save_figure(fig, title),
                               color=self.colors['accent'],
                               width=200, height=50)
        save_btn.pack(side=tk.RIGHT, padx=10)
        
        home_btn = ModernButton(btn_frame, "🏠 ACCUEIL",
                               self.create_main_dashboard,
                               color=self.colors['secondary'],
                               width=200, height=50)
        home_btn.pack(side=tk.RIGHT, padx=10)
    
    def show_valeurs_propres(self):
        """Valeurs propres et inerties"""
        self.show_loading("Calcul des valeurs propres...")
        
        def compute():
            try:
                if self.data_manager.df is None:
                    self.root.after(0, lambda: messagebox.showerror("Erreur", "Données non chargées"))
                    return
                
                if self.data_manager.pca is None:
                    self.data_manager.run_acp_full()
                
                vp = self.data_manager.pca_results['valeurs_propres']
                var_exp = self.data_manager.pca_results['variance_expliquee']
                var_cum = self.data_manager.pca_results['variance_cumulee']
                
                vp_df = pd.DataFrame({
                    'Composante': [f'PC{i+1}' for i in range(len(vp))],
                    'Valeur Propre': vp,
                    'Inertie (%)': var_exp,
                    'Inertie Cumulée (%)': var_cum
                })
                
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
                
                # Éboulis
                ax1.plot(range(1, len(vp)+1), vp, 'bo-', linewidth=2, markersize=8)
                ax1.axhline(y=1, color='r', linestyle='--', label='Kaiser (λ=1)')
                ax1.set_xlabel('Composante', fontsize=11, fontweight='bold')
                ax1.set_ylabel('Valeur Propre', fontsize=11, fontweight='bold')
                ax1.set_title('Scree Plot', fontsize=13, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend(fontsize=9)
                ax1.set_xticks(range(1, len(vp)+1))
                
                # Variance cumulée
                ax2.plot(range(1, len(var_cum)+1), var_cum, 'go-', linewidth=2, markersize=8)
                ax2.axhline(y=80, color='r', linestyle='--', label='Seuil 80%')
                ax2.fill_between(range(1, len(var_cum)+1), var_cum, alpha=0.3, color='green')
                ax2.set_xlabel('Composantes', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Variance (%)', fontsize=11, fontweight='bold')
                ax2.set_title('Variance Cumulée', fontsize=13, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend(fontsize=9)
                ax2.set_xticks(range(1, len(var_cum)+1))
                ax2.set_ylim([0, 105])
                
                # Critères
                ax3.axis('off')
                n_kaiser = np.sum(vp >= 1)
                n_80 = np.argmax(var_cum >= 80) + 1
                
                criteria_text = f"""🎯 CRITÈRES DE SÉLECTION:
                
• KAISER (λ≥1):
  Composantes: {n_kaiser}
  Variance: {var_cum[n_kaiser-1]:.2f}%

• RÈGLE 80%:
  Composantes: {n_80}
  Variance: {var_cum[n_80-1]:.2f}%

📈 RECOMMANDATION:
"""
                if n_kaiser <= 3:
                    criteria_text += f"✅ {n_kaiser} composantes (Kaiser)"
                else:
                    criteria_text += f"✅ {n_80} composantes (80%)"
                
                ax3.text(0.1, 0.5, criteria_text, fontsize=10, va='center',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
                
                plt.tight_layout()
                
                self.root.after(0, lambda: self.display_matplotlib_result(
                    "VALEURS PROPRES ET INERTIES", fig, "Analyse des composantes"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_qualite_representation(self):
        """Qualité de représentation"""
        self.show_loading("Calcul de la qualité de représentation...")
        
        def compute():
            try:
                if self.data_manager.pca is None:
                    self.data_manager.run_acp_full()
                
                cos2 = self.data_manager.pca_results['cos2']
                n_kaiser = self.data_manager.pca_results['n_composantes_kaiser']
                
                cos2_df = pd.DataFrame(
                    cos2[:, :n_kaiser],
                    columns=[f'PC{i+1}' for i in range(n_kaiser)],
                    index=self.data_manager.df['Nom du Tunnel']
                )
                cos2_df['Total'] = cos2_df.sum(axis=1)
                
                fig, axes = plt.subplots(2, 2, figsize=(22, 20))
                fig.suptitle('QUALITÉ DE REPRÉSENTATION (COS²)', 
                            fontsize=18, fontweight='bold', y=0.995)
                
                # 4 heatmaps
                for idx, (start, end, ax, title) in enumerate([
                    (0, 25, axes[0,0], '1-25'),
                    (25, 50, axes[0,1], '26-50'),
                    (50, 75, axes[1,0], '51-75'),
                    (75, 100, axes[1,1], '76-100')
                ]):
                    sns.heatmap(cos2_df.iloc[start:end, :n_kaiser].round(2), 
                               annot=True, fmt='.2f', cmap='YlOrRd', center=0.5,
                               ax=ax, cbar=True, annot_kws={'size': 8},
                               linewidths=0.5, linecolor='white')
                    ax.set_title(f'TUNNELS {title}', fontsize=13, fontweight='bold', pad=10)
                    ax.set_xlabel('PC', fontsize=10)
                    ax.set_ylabel('Tunnels', fontsize=10)
                    ax.tick_params(axis='x', rotation=45, labelsize=8)
                    ax.tick_params(axis='y', labelsize=7)
                
                plt.tight_layout(rect=[0, 0, 1, 0.99])
                
                stats_text = f"📈 STATISTIQUES:\n"
                stats_text += f"Cos² moyen: {cos2_df['Total'].mean():.3f}\n"
                stats_text += f"Min: {cos2_df['Total'].min():.3f}\n"
                stats_text += f"Max: {cos2_df['Total'].max():.3f}"
                
                self.root.after(0, lambda: self.display_matplotlib_result(
                    "QUALITÉ DE REPRÉSENTATION", fig, stats_text))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_coordonnees_individus(self):
        """Coordonnées des individus"""
        self.show_loading("Calcul des coordonnées...")
        
        def compute():
            try:
                if self.data_manager.pca is None:
                    self.data_manager.run_acp_full()
                
                coords = self.data_manager.pca_results['coords_individus']
                
                # Afficher les 25 premiers
                sample_df = coords.iloc[:25].round(3)
                
                self.root.after(0, lambda: self.display_dataframe_result(
                    "COORDONNÉES INDIVIDUS (25 premiers)", sample_df,
                    f"Coordonnées sur {len(coords.columns)} composantes principales"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_plan_individus(self):
        """Plan factoriel individus"""
        self.show_loading("Génération du plan factoriel...")
        
        def compute():
            try:
                if self.data_manager.pca is None:
                    self.data_manager.run_acp_full()
                
                coords = self.data_manager.pca_results['coords_individus']
                var_exp = self.data_manager.pca_results['variance_expliquee']
                
                fig, ax = plt.subplots(figsize=(16, 14))
                
                scatter = ax.scatter(coords['PC1'], coords['PC2'], 
                                   c=self.data_manager.df['Durée exploitation (ans)'], 
                                   cmap='viridis', s=150, alpha=0.6, 
                                   edgecolors='black', linewidth=1.5)
                
                # Annoter les tunnels extrêmes
                distances = np.sqrt(coords['PC1']**2 + coords['PC2']**2)
                top_indices = distances.nlargest(15).index
                
                for idx in top_indices:
                    tunnel_name = self.data_manager.df.loc[idx, 'Nom du Tunnel']
                    x = coords.loc[idx, 'PC1']
                    y = coords.loc[idx, 'PC2']
                    ax.annotate(tunnel_name, (x, y),
                               fontsize=7, ha='right', va='bottom', alpha=0.8,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5))
                
                ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
                ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel(f'PC1 ({var_exp[0]:.2f}%)', fontsize=13, fontweight='bold')
                ax.set_ylabel(f'PC2 ({var_exp[1]:.2f}%)', fontsize=13, fontweight='bold')
                ax.set_title('Plan Factoriel - 100 TUNNELS', fontsize=15, fontweight='bold', pad=20)
                
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Durée (ans)', fontsize=11, fontweight='bold')
                
                plt.tight_layout()
                
                self.root.after(0, lambda: self.display_matplotlib_result(
                    "PLAN FACTORIEL INDIVIDUS", fig, "Projection PC1 × PC2"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_cercle_correlation(self):
        """Cercle de corrélation"""
        self.show_loading("Génération du cercle de corrélation...")
        
        def compute():
            try:
                if self.data_manager.pca is None:
                    self.data_manager.run_acp_full()
                
                var_exp = self.data_manager.pca_results['variance_expliquee']
                correlations = self.data_manager.pca_results['correlations']
                
                fig, ax = plt.subplots(figsize=(11, 11))
                
                cercle = plt.Circle((0,0), 1, color='navy', fill=False, linewidth=2.5)
                ax.add_patch(cercle)
                
                ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)
                ax.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(self.data_manager.variables)))
                
                for i, (var, color) in enumerate(zip(self.data_manager.variables, colors)):
                    x, y = correlations[i, 0], correlations[i, 1]
                    ax.arrow(0, 0, x, y, head_width=0.04, head_length=0.04, 
                            fc=color, ec=color, linewidth=2.5, alpha=0.8)
                    
                    offset = 1.18
                    ax.text(x*offset, y*offset, var.split('(')[0].strip(), 
                           fontsize=10, ha='center', va='center', 
                           fontweight='bold', color='darkblue',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
                
                ax.set_xlim(-1.3, 1.3)
                ax.set_ylim(-1.3, 1.3)
                ax.set_xlabel(f'PC1 ({var_exp[0]:.2f}%)', fontsize=13, fontweight='bold')
                ax.set_ylabel(f'PC2 ({var_exp[1]:.2f}%)', fontsize=13, fontweight='bold')
                ax.set_title('Cercle de Corrélation', fontsize=15, fontweight='bold', pad=20)
                ax.set_aspect('equal')
                
                plt.tight_layout()
                
                self.root.after(0, lambda: self.display_matplotlib_result(
                    "CERCLE DE CORRÉLATION", fig, "Variables proches = corrélées"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_kmeans(self, n_clusters):
        """K-Means clustering"""
        self.show_loading(f"Clustering K={n_clusters}...")
        
        def compute():
            try:
                if self.data_manager.X_scaled is None:
                    self.data_manager.run_acp_full()
                
                if f'K{n_clusters}' not in self.data_manager.kmeans_models:
                    self.data_manager.run_kmeans(n_clusters)
                
                model_data = self.data_manager.kmeans_models[f'K{n_clusters}']
                
                # Plan factoriel avec clusters
                if self.data_manager.pca is None:
                    self.data_manager.run_acp_full()
                
                coords = self.data_manager.pca_results['coords_individus']
                clusters = model_data['clusters']
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                colors_k = ['red', 'blue', 'green', 'orange'][:n_clusters]
                for i in range(n_clusters):
                    mask = clusters == i
                    ax1.scatter(coords['PC1'][mask], coords['PC2'][mask],
                              c=colors_k[i], label=f'Cluster {i}', s=100, alpha=0.6, 
                              edgecolors='black', linewidth=1.5)
                
                # Annoter quelques tunnels
                for i in range(min(15, len(self.data_manager.df))):
                    tunnel_name = self.data_manager.df.iloc[i]['Nom du Tunnel']
                    ax1.annotate(tunnel_name, 
                                (coords.iloc[i]['PC1'], coords.iloc[i]['PC2']), 
                                fontsize=6, ha='right', va='bottom', alpha=0.7)
                
                centres_pca = self.data_manager.pca.transform(model_data['model'].cluster_centers_)
                for i in range(n_clusters):
                    ax1.scatter(centres_pca[i, 0], centres_pca[i, 1],
                              c=colors_k[i], marker='X', s=400, 
                              edgecolors='black', linewidth=3, zorder=5)
                
                ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
                ax1.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlabel('PC1', fontsize=12, fontweight='bold')
                ax1.set_ylabel('PC2', fontsize=12, fontweight='bold')
                ax1.set_title(f'K-Means K={n_clusters}', fontsize=14, fontweight='bold')
                ax1.legend(fontsize=10)
                
                # Heatmap centres
                centres_df = pd.DataFrame(
                    model_data['model'].cluster_centers_,
                    columns=[v.split('(')[0].strip() for v in self.data_manager.variables],
                    index=[f'C{i}' for i in range(n_clusters)]
                )
                
                sns.heatmap(centres_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                           ax=ax2, linewidths=0.5, annot_kws={'size': 9})
                ax2.set_title(f'Centres K={n_clusters}', fontsize=13, fontweight='bold')
                ax2.tick_params(axis='x', rotation=45, labelsize=9)
                
                plt.tight_layout()
                
                analysis_text = f"""K={n_clusters} CLUSTERS
                
MÉTRIQUES:
• Silhouette: {model_data['silhouette']:.4f}
• Davies-Bouldin: {model_data['davies']:.4f}
• Calinski: {model_data['calinski']:.2f}
"""
                
                self.root.after(0, lambda: self.display_matplotlib_result(
                    f"K-Means K={n_clusters}", fig, analysis_text))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_kmeans_comparison(self):
        """Comparaison K=3 vs K=4"""
        self.show_loading("Comparaison K=3 vs K=4...")
        
        def compute():
            try:
                if self.data_manager.X_scaled is None:
                    self.data_manager.run_acp_full()
                
                for k in [3, 4]:
                    if f'K{k}' not in self.data_manager.kmeans_models:
                        self.data_manager.run_kmeans(k)
                
                k3 = self.data_manager.kmeans_models['K3']
                k4 = self.data_manager.kmeans_models['K4']
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Métriques comparées
                x = range(2)
                k_values = [3, 4]
                
                silhouette_vals = [k3['silhouette'], k4['silhouette']]
                davies_vals = [k3['davies'], k4['davies']]
                calinski_vals = [k3['calinski'], k4['calinski']]
                
                ax1.bar(x, silhouette_vals, color=['#3b82f6', '#10b981'], alpha=0.7)
                ax1.set_xticks(x)
                ax1.set_xticklabels(['K=3', 'K=4'])
                ax1.set_ylabel('Score', fontsize=10, fontweight='bold')
                ax1.set_title('Silhouette Score', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='y')
                
                for i, val in enumerate(silhouette_vals):
                    ax1.text(i, val + 0.01, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
                
                ax2.bar(x, davies_vals, color=['#3b82f6', '#10b981'], alpha=0.7)
                ax2.set_xticks(x)
                ax2.set_xticklabels(['K=3', 'K=4'])
                ax2.set_ylabel('Score', fontsize=10, fontweight='bold')
                ax2.set_title('Davies-Bouldin (↓ mieux)', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                
                for i, val in enumerate(davies_vals):
                    ax2.text(i, val + 0.05, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
                
                ax3.bar(x, calinski_vals, color=['#3b82f6', '#10b981'], alpha=0.7)
                ax3.set_xticks(x)
                ax3.set_xticklabels(['K=3', 'K=4'])
                ax3.set_ylabel('Score', fontsize=10, fontweight='bold')
                ax3.set_title('Calinski-Harabasz (↑ mieux)', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='y')
                
                for i, val in enumerate(calinski_vals):
                    ax3.text(i, val + 50, f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')
                
                # Distribution clusters
                ax4.axis('off')
                dist_text = "📊 DISTRIBUTION CLUSTERS:\n"
                dist_text += "="*30 + "\n\n"
                
                for k, model_data in [(3, k3), (4, k4)]:
                    clusters = model_data['clusters']
                    unique, counts = np.unique(clusters, return_counts=True)
                    
                    dist_text += f"K={k}:\n"
                    for cluster_id, count in zip(unique, counts):
                        percent = count / len(clusters) * 100
                        dist_text += f"  Cluster {cluster_id}: {count} tunnels ({percent:.1f}%)\n"
                    dist_text += "\n"
                
                dist_text += f"RECOMMANDATION: "
                if k3['silhouette'] > k4['silhouette']:
                    dist_text += "K=3 (meilleure silhouette)"
                else:
                    dist_text += "K=4 (plus de granularité)"
                
                ax4.text(0.1, 0.5, dist_text, fontsize=9, va='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
                
                plt.tight_layout()
                
                self.root.after(0, lambda: self.display_matplotlib_result(
                    "Comparaison K=3 vs K=4", fig, "Analyse comparative"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_random_forest(self, data_type):
        """Random Forest"""
        self.show_loading(f"Random Forest {data_type}...")
        
        def compute():
            try:
                if data_type not in self.data_manager.rf_models:
                    self.data_manager.run_random_forest(data_type)
                
                results = self.data_manager.rf_models[data_type]
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Prédictions vs Réelles
                ax1.scatter(results['y_test'], results['y_test_pred'], 
                           alpha=0.6, s=80, edgecolors='black', linewidth=1)
                min_val = min(results['y_test'].min(), results['y_test_pred'].min())
                max_val = max(results['y_test'].max(), results['y_test_pred'].max())
                ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                ax1.set_xlabel('Valeurs Réelles', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Valeurs Prédites', fontsize=12, fontweight='bold')
                ax1.set_title('Prédictions vs Réelles', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                # Importance des variables
                if results['feature_importance'] is not None:
                    top_features = results['feature_importance'].head(10)
                    ax2.barh(range(len(top_features)), top_features['Importance'].values)
                    ax2.set_yticks(range(len(top_features)))
                    ax2.set_yticklabels(top_features['Feature'].values, fontsize=9)
                    ax2.set_xlabel('Importance', fontsize=12, fontweight='bold')
                    ax2.set_title('Top 10 Features Importantes', fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                
                # Métriques
                ax3.axis('off')
                metrics_text = f"""📊 MÉTRIQUES ({data_type.upper()})
                
ENSEMBLE DE TEST:
• R² Score: {results['test_r2']:.4f}
• RMSE: {results['test_rmse']:.4f}
• MAE: {results['test_mae']:.4f}

ENSEMBLE D'ENTRAÎNEMENT:
• R² Score: {results['train_r2']:.4f}
• RMSE: {results['train_rmse']:.4f}
• MAE: {results['train_mae']:.4f}
"""
                ax3.text(0.1, 0.5, metrics_text, fontsize=10, va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
                
                # Histogramme des erreurs
                errors = results['y_test'] - results['y_test_pred']
                ax4.hist(errors, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
                ax4.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {errors.mean():.2f}')
                ax4.set_xlabel('Erreur de prédiction', fontsize=12, fontweight='bold')
                ax4.set_ylabel('Fréquence', fontsize=12, fontweight='bold')
                ax4.set_title('Distribution des erreurs', fontsize=14, fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                self.root.after(0, lambda: self.display_matplotlib_result(
                    f"Random Forest ({data_type})", fig, 
                    f"Analyse RF sur données {data_type}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_afc_discretisation(self):
        """Discrétisation AFC"""
        self.show_loading("Discrétisation des variables...")
        
        def compute():
            try:
                if not self.data_manager.afc_results:
                    self.data_manager.run_afc_analysis()
                
                discrete_df = self.data_manager.afc_results['discrete_df']
                
                distribution_text = "DISCRÉTISATION - 8 VARIABLES:\n"
                distribution_text += "="*50 + "\n\n"
                distribution_text += "3 catégories: B (Bas) | M (Moyen) | H (Haut)\n\n"
                
                for col in discrete_df.columns:
                    counts = discrete_df[col].value_counts().sort_index()
                    total = counts.sum()
                    distribution_text += f"\n{col}:\n"
                    for cat, count in counts.items():
                        distribution_text += f"  {cat}: {count} ({count/total*100:.1f}%)\n"
                
                self.root.after(0, lambda: self.display_text_result(
                    "AFC - DISCRÉTISATION", distribution_text, "Préparation AFC"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_afc_tableau(self):
        """Tableau disjonctif AFC"""
        self.show_loading("Génération du tableau disjonctif...")
        
        def compute():
            try:
                if not self.data_manager.afc_results:
                    self.data_manager.run_afc_analysis()
                
                tableau = self.data_manager.afc_results['tableau_disjonctif']
                
                # Afficher un échantillon
                sample = tableau.head(20).astype(int)
                
                self.root.after(0, lambda: self.display_dataframe_result(
                    "TABLEAU DISJONCTIF AFC (20 premiers)", sample,
                    f"Dimensions complètes: {tableau.shape[0]}×{tableau.shape[1]}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_afc_plan(self):
        """Plan factoriel AFC"""
        self.show_loading("Génération du plan factoriel AFC...")
        
        def compute():
            try:
                if not self.data_manager.afc_results:
                    self.data_manager.run_afc_analysis()
                
                coords_individus = self.data_manager.afc_results['coords_individus']
                coords_modalites = self.data_manager.afc_results['coords_modalites']
                inertie_expl = self.data_manager.afc_results['inertie_expl']
                
                fig, ax = plt.subplots(figsize=(14, 10))
                
                # Individus (modalités)
                scatter1 = ax.scatter(coords_individus['F1'], coords_individus['F2'], 
                                     color='blue', label='Modalités', s=80, alpha=0.8)
                
                # Variables
                scatter2 = ax.scatter(coords_modalites['F1'], coords_modalites['F2'], 
                                     color='red', marker='s', label='Variables', s=200, 
                                     edgecolors='darkred', linewidth=1.5)
                
                # Annotations
                for i in range(min(10, len(coords_individus))):
                    ax.text(coords_individus['F1'].iloc[i] + 0.02, 
                           coords_individus['F2'].iloc[i] + 0.02, 
                           f"M{i+1}", color='blue', fontsize=9, fontweight='bold')
                
                for i in range(min(8, len(coords_modalites))):
                    ax.text(coords_modalites['F1'].iloc[i] + 0.03, 
                           coords_modalites['F2'].iloc[i] + 0.03, 
                           f"V{i+1}", color='red', fontsize=9, fontweight='bold')
                
                ax.axhline(0, color='black', linewidth=2)
                ax.axvline(0, color='black', linewidth=2)
                ax.set_xlabel(f'F1 ({inertie_expl[0]:.1f}%)', fontsize=14, fontweight='bold')
                ax.set_ylabel(f'F2 ({inertie_expl[1]:.1f}%)', fontsize=14, fontweight='bold')
                ax.set_title(f'PLAN FACTORIEL AFC\nInertie F1+F2: {inertie_expl[:2].sum():.1f}%', 
                           fontsize=16, fontweight='bold', pad=20)
                ax.legend(fontsize=12, loc='upper right')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                
                self.root.after(0, lambda: self.display_matplotlib_result(
                    "PLAN FACTORIEL AFC", fig, "Projection F1 × F2"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_isolation_forest(self):
        """Isolation Forest"""
        self.show_loading("Détection anomalies (Isolation Forest)...")
        
        def compute():
            try:
                if not self.data_manager.cyber_results:
                    self.data_manager.run_cyber_analysis()
                
                iso_pred = self.data_manager.cyber_results['iso_pred']
                n_anomalies = (iso_pred == -1).sum()
                n_normaux = (iso_pred == 1).sum()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                colors_if = ['red' if p == -1 else 'green' for p in iso_pred]
                ax1.scatter(self.data_manager.X.iloc[:, 0], 
                          self.data_manager.X.iloc[:, 1],
                          c=colors_if, s=80, alpha=0.6, edgecolors='black')
                
                ax1.set_xlabel(self.data_manager.variables[0], fontsize=11, fontweight='bold')
                ax1.set_ylabel(self.data_manager.variables[1], fontsize=11, fontweight='bold')
                ax1.set_title(f'Isolation Forest - {n_anomalies} anomalies', 
                            fontsize=13, fontweight='bold', pad=10)
                ax1.grid(True, alpha=0.3)
                
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='red', edgecolor='black', label=f'Anomalie ({n_anomalies})'),
                    Patch(facecolor='green', edgecolor='black', label=f'Normal ({n_normaux})')
                ]
                ax1.legend(handles=legend_elements, loc='best', fontsize=9)
                
                # Top anomalies
                ax2.axis('off')
                if n_anomalies > 0:
                    anomaly_indices = np.where(iso_pred == -1)[0]
                    risk_text = f"🚨 ANOMALIES DÉTECTÉES ({n_anomalies}):\n{'='*35}\n\n"
                    
                    for i, idx in enumerate(anomaly_indices[:10], 1):
                        tunnel_name = self.data_manager.df.iloc[idx]['Nom du Tunnel']
                        risk_text += f"{i}. {tunnel_name}\n"
                    
                    if n_anomalies > 10:
                        risk_text += f"\n... +{n_anomalies-10} autres"
                else:
                    risk_text = "✅ Aucune anomalie détectée"
                
                ax2.text(0.1, 0.5, risk_text, fontsize=10, va='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
                
                plt.tight_layout()
                
                self.root.after(0, lambda: self.display_matplotlib_result(
                    "Isolation Forest", fig, f"{n_anomalies} anomalies détectées"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_lof(self):
        """Local Outlier Factor"""
        self.show_loading("Détection anomalies (LOF)...")
        
        def compute():
            try:
                if not self.data_manager.cyber_results:
                    self.data_manager.run_cyber_analysis()
                
                lof_pred = self.data_manager.cyber_results['lof_pred']
                n_anomalies = (lof_pred == -1).sum()
                n_normaux = (lof_pred == 1).sum()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                colors_lof = ['red' if p == -1 else 'green' for p in lof_pred]
                ax1.scatter(self.data_manager.X.iloc[:, 2], 
                          self.data_manager.X.iloc[:, 3],
                          c=colors_lof, s=80, alpha=0.6, edgecolors='black')
                
                ax1.set_xlabel(self.data_manager.variables[2], fontsize=11, fontweight='bold')
                ax1.set_ylabel(self.data_manager.variables[3], fontsize=11, fontweight='bold')
                ax1.set_title(f'LOF - {n_anomalies} anomalies', 
                            fontsize=13, fontweight='bold', pad=10)
                ax1.grid(True, alpha=0.3)
                
                # Top anomalies
                ax2.axis('off')
                if n_anomalies > 0:
                    anomaly_indices = np.where(lof_pred == -1)[0]
                    risk_text = f"🚨 ANOMALIES DÉTECTÉES ({n_anomalies}):\n{'='*35}\n\n"
                    
                    for i, idx in enumerate(anomaly_indices[:10], 1):
                        tunnel_name = self.data_manager.df.iloc[idx]['Nom du Tunnel']
                        risk_text += f"{i}. {tunnel_name}\n"
                    
                    if n_anomalies > 10:
                        risk_text += f"\n... +{n_anomalies-10} autres"
                else:
                    risk_text = "✅ Aucune anomalie détectée"
                
                ax2.text(0.1, 0.5, risk_text, fontsize=10, va='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
                
                plt.tight_layout()
                
                self.root.after(0, lambda: self.display_matplotlib_result(
                    "Local Outlier Factor", fig, f"{n_anomalies} anomalies"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    def show_cyber_comparison(self):
        """Comparaison IF vs LOF"""
        self.show_loading("Comparaison IF vs LOF...")
        
        def compute():
            try:
                if not self.data_manager.cyber_results:
                    self.data_manager.run_cyber_analysis()
                
                iso_pred = self.data_manager.cyber_results['iso_pred']
                lof_pred = self.data_manager.cyber_results['lof_pred']
                
                n_iso = (iso_pred == -1).sum()
                n_lof = (lof_pred == -1).sum()
                n_both = ((iso_pred == -1) & (lof_pred == -1)).sum()
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
                
                # Comparaison
                methods = ['IF', 'LOF']
                counts = [n_iso, n_lof]
                colors = ['#3b82f6', '#10b981']
                
                ax1.bar(methods, counts, color=colors, edgecolor='black', alpha=0.7)
                ax1.set_ylabel('Anomalies', fontsize=10, fontweight='bold')
                ax1.set_title('Comparaison IF vs LOF', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='y')
                
                for i, count in enumerate(counts):
                    ax1.text(i, count + 1, str(count), ha='center', fontweight='bold', fontsize=10)
                
                # Venn
                only_iso = n_iso - n_both
                only_lof = n_lof - n_both
                both = n_both
                
                labels = ['IF seul', 'LOF seul', 'Consensus']
                sizes = [only_iso, only_lof, both]
                colors_venn = ['#3b82f6', '#10b981', '#f59e0b']
                
                ax2.pie(sizes, labels=labels, colors=colors_venn, autopct='%1.1f%%',
                       startangle=90, wedgeprops={'edgecolor': 'black'}, textprops={'fontsize': 9})
                ax2.set_title('Chevauchement', fontsize=12, fontweight='bold')
                
                # Liste critiques
                ax3.axis('off')
                if n_both > 0:
                    crit_indices = np.where((iso_pred == -1) & (lof_pred == -1))[0]
                    crit_text = f"⚠️ CRITIQUES (Consensus):\n{'='*30}\n\n"
                    crit_text += f"Total: {n_both}\n\n"
                    
                    for i, idx in enumerate(crit_indices[:8]):
                        tunnel_name = self.data_manager.df.iloc[idx]['Nom du Tunnel']
                        crit_text += f"• {tunnel_name}\n"
                    
                    if n_both > 8:
                        crit_text += f"\n... +{n_both-8} autres"
                else:
                    crit_text = "✅ Aucun tunnel critique détecté par les deux méthodes"
                
                ax3.text(0.1, 0.5, crit_text, fontsize=10, va='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
                
                # Accord entre méthodes
                ax4.axis('off')
                accord = (iso_pred == lof_pred).sum()
                pct_accord = (accord / len(iso_pred)) * 100
                
                accord_text = f"📊 ACCORD ENTRE MÉTHODES:\n{'='*30}\n\n"
                accord_text += f"Accord: {pct_accord:.1f}%\n"
                accord_text += f"Désaccord: {100-pct_accord:.1f}%\n\n"
                
                if pct_accord > 70:
                    accord_text += "✅ Bon accord entre IF et LOF"
                elif pct_accord > 50:
                    accord_text += "⚠️ Accord modéré"
                else:
                    accord_text += "❌ Faible accord"
                
                ax4.text(0.1, 0.5, accord_text, fontsize=10, va='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
                
                plt.tight_layout()
                
                info_text = f"""COMPARAISON:

IF: {n_iso} anomalies
LOF: {n_lof} anomalies
Consensus: {n_both} tunnels
"""
                
                self.root.after(0, lambda: self.display_matplotlib_result(
                    "Comparaison IF vs LOF", fig, info_text))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur: {str(e)}"))
        
        threading.Thread(target=compute, daemon=True).start()
    
    # ==================== MÉTHODES UTILITAIRES ====================
    
    def load_data_file(self):
        """Charge le fichier de données"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner le fichier CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.show_loading("Chargement des données...")
            
            def load_thread():
                try:
                    if self.data_manager.load_data(file_path):
                        self.root.after(0, lambda: messagebox.showinfo("Succès", 
                                   f"Données chargées avec succès!\n"
                                   f"• {len(self.data_manager.df)} tunnels\n"
                                   f"• {len(self.data_manager.variables)} variables"))
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Erreur", "Impossible de charger le fichier"))
                finally:
                    self.root.after(0, self.create_main_dashboard)
            
            threading.Thread(target=load_thread, daemon=True).start()
    
    def show_loading(self, message="Calcul..."):
        """Écran de chargement"""
        self.clear_window()
        
        loading_frame = tk.Frame(self.root, bg=self.colors['dark'])
        loading_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(loading_frame, bg=self.colors['dark'], 
                          highlightthickness=0, width=200, height=200)
        canvas.pack(pady=150)
        
        self.loading_circle = canvas.create_oval(20, 20, 180, 180, 
                                                outline=self.colors['accent'], 
                                                width=8, fill="")
        
        tk.Label(loading_frame, text=message, 
                font=("Segoe UI", 16),
                bg=self.colors['dark'], fg=self.colors['white']).pack()
        
        tk.Label(loading_frame, text="Veuillez patienter...", 
                font=("Segoe UI", 12),
                bg=self.colors['dark'], fg="#6b7280").pack(pady=10)
        
        self.rotate_circle(canvas, 0)
    
    def rotate_circle(self, canvas, angle):
        """Animation rotation"""
        try:
            canvas.delete("loading_line")
            x = 100 + 80 * np.cos(np.radians(angle))
            y = 100 + 80 * np.sin(np.radians(angle))
            canvas.create_line(100, 100, x, y, 
                              fill=self.colors['accent'], 
                              width=8, tag="loading_line")
            self.root.after(20, lambda: self.rotate_circle(canvas, (angle + 10) % 360))
        except:
            pass
    
    def display_matplotlib_result(self, title, fig, description=""):
        """Affiche résultat matplotlib"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg=self.colors['dark'])
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['primary'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        back_btn = ModernButton(header_frame, "← RETOUR", 
                               lambda: self.return_to_module(self.current_page),
                               color="#374151", hover_color="#1f2937",
                               width=150, height=40)
        back_btn.place(x=20, y=20)
        
        tk.Label(header_frame, text=title,
                font=("Segoe UI", 18, "bold"),
                bg=self.colors['primary'], fg='white').pack(pady=25)
        
        # Contenu
        content_frame = tk.Frame(main_frame, bg='#1f2937')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Canvas matplotlib
        canvas = FigureCanvasTkAgg(fig, master=content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Description
        if description:
            desc_frame = tk.Frame(content_frame, bg='#1f2937')
            desc_frame.pack(fill=tk.X, pady=(10, 0))
            
            desc_text = scrolledtext.ScrolledText(desc_frame, 
                                                 font=("Segoe UI", 10),
                                                 bg='#1f2937', 
                                                 fg='white',
                                                 height=6)
            desc_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            desc_text.insert('1.0', description)
            desc_text.config(state='disabled')
        
        # Boutons
        btn_frame = tk.Frame(content_frame, bg='#1f2937')
        btn_frame.pack(fill=tk.X, pady=10)
        
        save_btn = ModernButton(btn_frame, "💾 SAUVEGARDER",
                               lambda: self.save_figure(fig, title),
                               color=self.colors['accent'],
                               width=200, height=50)
        save_btn.pack(side=tk.RIGHT, padx=10)
        
        home_btn = ModernButton(btn_frame, "🏠 ACCUEIL",
                               self.create_main_dashboard,
                               color=self.colors['secondary'],
                               width=200, height=50)
        home_btn.pack(side=tk.RIGHT, padx=10)
    
    def display_dataframe_result(self, title, dataframe, description=""):
        """Affiche dataframe"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg=self.colors['dark'])
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['primary'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        back_btn = ModernButton(header_frame, "← RETOUR", 
                               lambda: self.return_to_module(self.current_page),
                               color="#374151", hover_color="#1f2937",
                               width=150, height=40)
        back_btn.place(x=20, y=20)
        
        tk.Label(header_frame, text=title,
                font=("Segoe UI", 18, "bold"),
                bg=self.colors['primary'], fg='white').pack(pady=25)
        
        # Contenu
        content_frame = tk.Frame(main_frame, bg='#1f2937')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tree_frame = tk.Frame(content_frame, bg='#1f2937')
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        tree = ttk.Treeview(tree_frame, 
                           yscrollcommand=vsb.set, 
                           xscrollcommand=hsb.set,
                           style="Custom.Treeview")
        
        # Style personnalisé
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Treeview",
                       background='#1f2937',
                       foreground='white',
                       fieldbackground='#1f2937',
                       rowheight=25)
        style.configure("Custom.Treeview.Heading",
                       background=self.colors['primary'],
                       foreground='white',
                       relief='flat')
        
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)
        
        tree["columns"] = list(dataframe.columns)
        tree["show"] = "headings"
        
        for col in dataframe.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor='center')
        
        for i, row in dataframe.iterrows():
            tree.insert("", "end", values=list(row))
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        
        if description:
            desc_frame = tk.Frame(content_frame, bg='#1f2937')
            desc_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
            
            desc_text = scrolledtext.ScrolledText(desc_frame, 
                                                 font=("Segoe UI", 10),
                                                 bg='#1f2937', 
                                                 fg='white')
            desc_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            desc_text.insert('1.0', description)
            desc_text.config(state='disabled')
        
        btn_frame = tk.Frame(content_frame, bg='#1f2937')
        btn_frame.pack(fill=tk.X, pady=10)
        
        home_btn = ModernButton(btn_frame, "🏠 ACCUEIL",
                               self.create_main_dashboard,
                               color=self.colors['secondary'],
                               width=200, height=50)
        home_btn.pack(side=tk.RIGHT, padx=10)
    
    def display_text_result(self, title, text, description=""):
        """Affiche texte"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg=self.colors['dark'])
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['primary'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        back_btn = ModernButton(header_frame, "← RETOUR", 
                               lambda: self.return_to_module(self.current_page),
                               color="#374151", hover_color="#1f2937",
                               width=150, height=40)
        back_btn.place(x=20, y=20)
        
        tk.Label(header_frame, text=title,
                font=("Segoe UI", 18, "bold"),
                bg=self.colors['primary'], fg='white').pack(pady=25)
        
        # Contenu
        content_frame = tk.Frame(main_frame, bg='#1f2937')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        text_widget = scrolledtext.ScrolledText(content_frame,
                                              font=("Segoe UI", 11),
                                              bg='#1f2937',
                                              fg='white')
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert('1.0', text)
        text_widget.config(state='disabled')
        
        if description:
            desc_frame = tk.Frame(content_frame, bg='#1f2937')
            desc_frame.pack(fill=tk.X, pady=(10, 0))
            
            desc_label = tk.Label(desc_frame, text=description,
                                 font=("Segoe UI", 10, "italic"),
                                 bg='#1f2937', fg="#9ca3af")
            desc_label.pack(pady=5)
        
        btn_frame = tk.Frame(content_frame, bg='#1f2937')
        btn_frame.pack(fill=tk.X, pady=10)
        
        home_btn = ModernButton(btn_frame, "🏠 ACCUEIL",
                               self.create_main_dashboard,
                               color=self.colors['secondary'],
                               width=200, height=50)
        home_btn.pack(side=tk.RIGHT, padx=10)
    
    def return_to_module(self, module_name):
        """Retour au module"""
        if module_name == "acp":
            self.open_acp_dashboard()
        elif module_name == "ai":
            self.open_ai_dashboard()
        elif module_name == "afc":
            self.open_afc_dashboard()
        elif module_name == "cyber":
            self.open_cyber_dashboard()
        else:
            self.create_main_dashboard()
    
    def save_figure(self, fig, title):
        """Sauvegarde figure"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                      ("SVG files", "*.svg"), ("All files", "*.*")],
            initialfile=f"{title.replace(' ', '_')}.png"
        )
        
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Succès", f"Figure sauvegardée:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur sauvegarde: {str(e)}")
    
    def darken_color(self, color, percent):
        """Assombrit une couleur hexadécimale"""
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple(max(0, int(c * (100 - percent) / 100)) for c in rgb)
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

# Lancement application
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = TunnelAnalysisApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()