import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Configuration Streamlit
st.set_page_config(page_title="Détection Anomalies LVMH", layout="wide")

st.title("Détection d'Anomalies Financières - LVMH")
st.markdown("Projet IA pour la Finance | Analyse des séries temporelles")
st.markdown("Etudiant : Julie Testu & Quentin Tajchner")
st.divider()

# Chargement des données
@st.cache_data 
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df['Returns'] = df['clot'].pct_change()
        
        # Feature Engineering (Volatilité)
        volatility_window = 30
        df['Volatility'] = df['Returns'].rolling(window=volatility_window).std().shift(1)
        df['Volatility'] = df['Volatility'].bfill()
        df['Volatility'] = df['Volatility'].replace(0, 0.005)
        df['Standardized_Returns'] = df['Returns'] / df['Volatility']
        
        return df.dropna()
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        return None

# Chargement des inputs Excel
uploaded_file = st.sidebar.file_uploader("Charger le fichier Excel", type=["xlsx", "csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    # Chemin par défaut des inputs
    default_path = "Input_projet_LVMH.xlsx"
    try:
        df = load_data(default_path)
    except:
        st.warning("Veuillez charger le fichier Excel 'Input_projet_LVMH.xlsx' via le menu de gauche.")
        df = None

if df is not None:
    # Bar latérale
    st.sidebar.header("Paramètres du Modèle")   
    
    model_choice = st.sidebar.selectbox(
        "Choisir le Modèle IA",
        ("Isolation Forest", "One-Class SVM", "Local Outlier Factor")
    )
    
    contamination = st.sidebar.slider("Sensibilité (Contamination)", 0.001, 0.05, 0.01, 0.001)
    cooldown = st.sidebar.slider("Filtre Anti-Réplique (Jours)", 0, 30, 10)
    
    # Préparation du modèle
    data_model = df[['Standardized_Returns', 'clot', 'Returns']].copy()
    
    if model_choice == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=42)
    elif model_choice == "One-Class SVM":
        model = OneClassSVM(nu=contamination, kernel="rbf", gamma='scale')
    elif model_choice == "Local Outlier Factor":
        model = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)

    # Entraînement et Prédiction
    model.fit(data_model[['Standardized_Returns']])
    data_model['Anomaly'] = model.predict(data_model[['Standardized_Returns']])
    
    # Score
    if hasattr(model, "decision_function"):
        data_model['Score'] = model.decision_function(data_model[['Standardized_Returns']])
    else:
        data_model['Score'] = 0

    # Règles de secours
    seuil_max = 0.125
    data_model.loc[abs(data_model['Returns']) > seuil_max, 'Anomaly'] = -1
    
    seuil_min = 0.2
    data_model.loc[(data_model['Anomaly'] == -1) & (abs(data_model['Returns']) < seuil_min), 'Anomaly'] = 1

    # Filtrage anti-doublons
    raw_outliers = data_model[data_model['Anomaly'] == -1]
    
    kept_indices = []
    if not raw_outliers.empty:
        kept_indices = [raw_outliers.index[0]]
        last_date = raw_outliers.index[0]
        for current_date in raw_outliers.index[1:]:
            if (current_date - last_date).days > cooldown:
                kept_indices.append(current_date)
                last_date = current_date
    
    outliers = raw_outliers.loc[kept_indices] if kept_indices else raw_outliers.iloc[0:0]

    # Affichage des résultats
    col1, col2, col3 = st.columns(3)
    col1.metric("Anomalies Détectées", len(outliers))
    col2.metric("Modèle Utilisé", model_choice)
    col3.metric("Sensibilité", f"{contamination*100}%")

    # Graphiques
    st.subheader("1. Analyse du Cours et des Alertes")
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(df.index, df['clot'], label='Cours LVMH', color='slategray', alpha=0.8)
    ax1.scatter(outliers.index, outliers['clot'], color='crimson', label='Anomalies Validées', s=60, zorder=5)
    ax1.set_title("Cours de l'action avec anomalies détectées")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("2. Rendements (Volatilité)")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(df.index, df['Returns'], color='darkgray', alpha=0.5)
        ax2.scatter(outliers.index, outliers['Returns'], color='crimson', s=40)
        ax2.set_title("Variations journalières (%)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    with col_g2:
        st.subheader("3. Score de l'IA")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(data_model.index, data_model['Score'], color='teal')
        ax3.scatter(outliers.index, outliers['Score'], color='crimson', s=40)
        ax3.axhline(0, color='black', linestyle='--')
        ax3.set_title("Score de décision ( < 0 = Anomalie)")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        
    # Analyse statistique
    st.divider()
    st.subheader("4. Analyse Statistique et Distribution")
    
    col_stat1, col_stat2 = st.columns(2)

    # Boxplot comparatif
    with col_stat1:
        st.markdown("**Comparaison des amplitudes (Normal vs Anomalie)**")
        fig_box, ax_box = plt.subplots(figsize=(8, 6))
        
        plot_data = data_model.copy()
        plot_data['Type'] = plot_data['Anomaly'].apply(lambda x: 'Anomalie' if x == -1 else 'Normal')
        plot_data['Amplitude_Absolue'] = plot_data['Returns'].abs() * 100 
        
        sns.boxplot(x='Type', y='Amplitude_Absolue', data=plot_data, ax=ax_box,
                    palette={'Normal': 'seagreen', 'Anomalie': 'crimson'})
        
        ax_box.set_ylabel("Variation absolue (%)")
        ax_box.set_yscale('log')
        ax_box.grid(True, alpha=0.3)
        st.pyplot(fig_box)

    # Histogramme des cours
    with col_stat2:
        st.markdown("**Distribution des Cours de Clôture**")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
        
        mean_val = df['clot'].mean()
        median_val = df['clot'].median()
        
        sns.histplot(df['clot'], bins=50, kde=True, color='midnightblue', alpha=0.6, ax=ax_hist)
        
        ax_hist.axvline(mean_val, color='crimson', linestyle='--', linewidth=2, label=f"Moyenne ({mean_val:.0f}€)")
        ax_hist.axvline(median_val, color='teal', linestyle='-', linewidth=2, label=f"Médiane ({median_val:.0f}€)")
        
        ax_hist.set_xlabel("Prix (€)")
        ax_hist.set_ylabel("Fréquence")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        st.pyplot(fig_hist)

    # Ajout du tableau des anomalies
    st.divider()
    st.subheader(f"Liste détaillée des {len(outliers)} anomalies détectées")
    
    if not outliers.empty:
        # On prépare un dataframe propre pour l'affichage (renommage des colonnes)
        df_display = outliers[['clot', 'Returns', 'Score']].copy()
        df_display.columns = ['Prix Clôture (€)', 'Variation (%)', 'Score IA']
        
        # Affichage avec formatage
        st.dataframe(
            df_display.style.format({
                'Prix Clôture (€)': '{:.2f} €',
                'Variation (%)': '{:.2%}',
                'Score IA': '{:.4f}'
            }).background_gradient(subset=['Variation (%)'], cmap='Reds_r'),
            use_container_width=True
        )
        
        # Bouton de téléchargement CSV
        csv = df_display.to_csv().encode('utf-8')
        st.download_button(
            label="Télécharger les anomalies en CSV",
            data=csv,
            file_name='anomalies_lvmh.csv',
            mime='text/csv',
        )
    else:
        st.info("Aucune anomalie détectée avec les paramètres actuels.")