# Importation des bibliothèques :
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Chargement des inputs :
df = pd.read_excel(r"C:\Users\tajch\Documents\INGE 3\IA pour la finance\Input_projet_LVMH.xlsx")

# Préparation des données :
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()
df['Returns'] = df['clot'].pct_change()

volatility_window = 30
df['Volatility'] = df['Returns'].rolling(window=volatility_window).std().shift(1)

df['Volatility'] = df['Volatility'].bfill()

df['Volatility'] = df['Volatility'].replace(0, 0.001)

df['Standardized_Returns'] = df['Returns'] / df['Volatility']

data_model = df[['Standardized_Returns', 'clot', 'Returns']].dropna()

# Configuration des 3 modèles à entraîner :
models = {
    "Isolation Forest": IsolationForest(contamination=0.01, random_state=42),
    "One-Class SVM": OneClassSVM(nu=0.01, kernel="rbf", gamma='scale'),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
}

for model_name, model in models.items():
    print(f"\n Début du traitement pour : {model_name}")
    
    current_data_model = data_model.copy()

    # Entraînement du modèle
    model.fit(current_data_model[['Standardized_Returns']])

    # Prédiction & Score
    current_data_model['Anomaly'] = model.predict(current_data_model[['Standardized_Returns']])
    
    # Gestion du score
    if hasattr(model, "decision_function"):
        current_data_model['Score'] = model.decision_function(current_data_model[['Standardized_Returns']])
    else:
        current_data_model['Score'] = 0 # Cas de secours

    # Réitération en cas d'oublies
    seuil_absolu = 0.125
    current_data_model.loc[abs(current_data_model['Returns']) > seuil_absolu, 'Anomaly'] = -1

    # Filtre de suppression du bruit
    seuil_absolu_min = 0.2
    current_data_model.loc[(current_data_model['Anomaly'] == -1) & (abs(current_data_model['Returns']) < seuil_absolu_min), 'Anomaly'] = 1

    # Fusion des données au dataframe (On nettoie df avant de joindre pour éviter les doublons de colonnes)
    cols_to_join = ['Anomaly', 'Score']
    df_loop = df.drop(columns=[c for c in cols_to_join if c in df.columns], errors='ignore')
    df_loop = df_loop.join(current_data_model[cols_to_join], how='inner')

    raw_outliers = df_loop[df_loop['Anomaly'] == -1] # Récupération des anomalies brutes

    # Filtrage des doublons :
    def filter_consecutive_anomalies(outliers_df, days_cooldown=10):
        if outliers_df.empty:
            return outliers_df
        kept_indices = [outliers_df.index[0]]
        last_date = outliers_df.index[0]
        for current_date in outliers_df.index[1:]:
            delta = (current_date - last_date).days
            if delta > days_cooldown:
                kept_indices.append(current_date)
                last_date = current_date
        return outliers_df.loc[kept_indices]

    # On applique le filtre
    if not raw_outliers.empty:
        outliers = filter_consecutive_anomalies(raw_outliers, days_cooldown=10)
    else:
        outliers = raw_outliers

    print(f"[{model_name}] Anomalies détectées (Brut) : {len(raw_outliers)}")
    print(f"[{model_name}] Anomalies après filtrage : {len(outliers)}")

    # Visualisations graphiques :
    
    # 1. Histogramme des cours de clôture
    mean_clot = df_loop['clot'].mean()
    median_clot = df_loop['clot'].median()
    q1 = df_loop['clot'].quantile(0.25)
    q3 = df_loop['clot'].quantile(0.75)

    plt.figure(figsize=(10, 6))
    sns.histplot(df_loop['clot'], bins=50, kde=True, color='midnightblue', alpha=0.6)
    plt.axvline(mean_clot, color='red', linestyle=':', linewidth=2, label=f"Moyenne = {mean_clot:.2f}")
    plt.axvline(median_clot, color='black', linestyle='-', linewidth=2, label=f"Médiane = {median_clot:.2f}")
    plt.axvline(q1, color='green', linestyle=':', linewidth=2, label=f"Q1 = {q1:.2f}")
    plt.axvline(q3, color='orange', linestyle=':', linewidth=2, label=f"Q3 = {q3:.2f}")
    plt.title(f"[{model_name}] Histogramme des cours", fontsize=14)
    plt.xlim(-100, 1000)
    plt.xlabel("Cours de clôture")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. Graphique des anomalies détectées :
    plt.figure(figsize=(15, 6))
    plt.plot(df_loop.index, df_loop['clot'], label='Cours LVMH', color='slategray', alpha=0.8, linewidth=1)   
    plt.scatter(outliers.index, outliers['clot'], color='crimson', label='Anomalies', s=60, zorder=5)   
    ignored = raw_outliers.index.difference(outliers.index)
    plt.scatter(ignored, df_loop.loc[ignored]['clot'], color='gainsboro', s=40, label='Doublons ignorés', alpha=1, zorder=3)   
    plt.title(f"[{model_name}] Détection d'Anomalies (Sans doublons)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 3. Graphique des rendements :
    plt.figure(figsize=(15, 6))
    plt.plot(df_loop.index, df_loop['Returns'], label='Rendements journaliers', color='darkgray', alpha=0.5, linewidth=0.8)   
    plt.scatter(outliers.index, outliers['Returns'], color='crimson', label='Anomalies Détectées', s=50, zorder=5)
    plt.title(f"[{model_name}] Anomalies sur les Variations", fontsize=14)
    plt.ylabel("Variation (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 4. Graphique du score d'anomalie :
    plt.figure(figsize=(15, 6))
    plt.plot(df_loop.index, df_loop['Score'], color='teal', label='Score', linewidth=1.2)
    plt.scatter(outliers.index, outliers['Score'], color='crimson', s=30, label='Seuil franchi')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, label='Seuil Zéro')
    plt.title(f"[{model_name}] Score de l'algorithme", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 5. Boxplot comparatif :
    plt.figure(figsize=(8, 6))
    plot_data = current_data_model.copy()
    plot_data['Type'] = plot_data['Anomaly'].apply(lambda x: 'Anomalie' if x == -1 else 'Normal')
    plot_data['Amplitude_Absolue'] = plot_data['Returns'].abs() * 100 
    sns.boxplot(x='Type', y='Amplitude_Absolue', data=plot_data, palette={'Normal': 'seagreen', 'Anomalie': 'crimson'})
    plt.title(f"[{model_name}] Comparaison : Amplitude des variations", fontsize=14)
    plt.ylabel("Variation absolue journalière (%)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()