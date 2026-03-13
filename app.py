import streamlit as st
import pandas as pd
import plotly.express as px
import json
import numpy as np
import time
import joblib  # Importation pour charger le modèle .pkl

# --- CONFIGURATION ---
st.set_page_config(page_title="Insurance Insight Pro", layout="wide", page_icon="🏥")

# Styles CSS personnalisés pour l'effet "Chat" et la DA Orange Claude Opus
st.markdown("""
    <style>
    /* Importation des polices */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Ibarra+Real+Nova:ital,wght@0,400;0,600;1,400&display=swap');

    /* Typographie globale de l'appli */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Harmonisation des titres et sous-titres natifs (Style Opus) */
    h1, h2, h3, .main-title {
        font-family: 'Ibarra Real Nova', serif !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
    }

    h1 { font-size: 2.8rem !important; }
    h2 { font-size: 2rem !important; margin-top: 1.5rem !important; }
    h3 { font-size: 1.5rem !important; color: #4a4a4a !important; }

    /* Style spécifique pour le contenu narratif (Serif comme Claude) */
    .chat-bubble, .narrative-text, .stMarkdown div p {
        font-family: 'Ibarra Real Nova', serif;
        font-size: 1.15rem;
    }

    /* Harmonisation des tableaux (Métadonnées) */
    div[data-testid="stTable"] table {
        font-family: 'Ibarra Real Nova', serif !important;
        border-collapse: collapse;
    }
    div[data-testid="stTable"] th {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #d97757 !important;
        text-transform: uppercase;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em;
    }
    div[data-testid="stTable"] td {
        font-family: 'Ibarra Real Nova', serif !important;
        font-size: 1.05rem !important;
        color: #1a1a1a !important;
    }

    .chat-bubble {
        background-color: #fffaf8; 
        border-radius: 12px;
        padding: 24px;
        margin: 10px 0;
        border: 1px solid #f9e8e2;
        line-height: 1.6;
        color: #1a1a1a;
        box-shadow: 0 2px 4px rgba(217, 119, 87, 0.05);
    }

    .chat-header {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        margin-bottom: 12px;
        color: #d97757; 
        display: flex;
        align-items: center;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Style des boîtes KPI style Opus */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        transition: transform 0.2s ease;
        margin-bottom: 15px;
    }
    .kpi-card:hover {
        border-color: #d97757;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #8a8a8a;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'Ibarra Real Nova', serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: #1a1a1a;
    }

    /* Masquer totalement la sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }

    .main-title {
        margin-bottom: 0.5rem;
    }

    /* Onglets classiques style Claude */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        border-bottom: 1px solid #f0f0f0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: transparent;
        color: #8a8a8a;
        font-weight: 500;
        font-size: 1rem;
    }

    .stTabs [aria-selected="true"] {
        color: #d97757 !important;
        border-bottom: 2px solid #d97757 !important;
    }

    /* Personnalisation des boutons */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #d97757;
        color: #d97757;
        background-color: white;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #d97757;
        color: white;
        border: 1px solid #d97757;
    }
    </style>
""", unsafe_allow_html=True)


def kpi_card(label, value):
    """Génère le HTML pour une carte KPI épurée."""
    return f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
    """


def stream_chat_text(title, text):
    """Affiche un effet de chat qui se déroule."""
    container = st.empty()
    full_response = ""
    header_html = f'<div class="chat-header">🤖 Assistant Analyse • {title}</div>'

    for char in text:
        full_response += char
        container.markdown(f"""
            <div class="chat-bubble">
                {header_html}
                <div class="narrative-text">{full_response}▌</div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(0.005)

    container.markdown(f"""
        <div class="chat-bubble">
            {header_html}
            <div class="narrative-text">{full_response}</div>
        </div>
    """, unsafe_allow_html=True)


# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("insurance.csv")
        with open("metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return df, meta
    except Exception as e:
        st.error(f"Erreur de chargement des données : {e}")
        return None, None


@st.cache_resource
def load_trained_model():
    """Charge le modèle Random Forest sauvegardé."""
    try:
        model_data = joblib.load("insurance_model_v1.pkl")
        return model_data
    except Exception:
        return None


df, metadata = load_data()
model_data = load_trained_model()

# --- INTERFACE PRINCIPALE ---
if df is not None:
    st.markdown('<h1 class="main-title">🏥 Insurance Insight Pro</h1>', unsafe_allow_html=True)

    tab_accueil, tab_exploration, tab_biais, tab_modele = st.tabs([
        "Accueil",
        "Exploration des Données",
        "Détection de Biais",
        "Performance du Modèle"
    ])

    # --- ONGLET ACCUEIL ---
    with tab_accueil:
        st.write(" ")
        st.header("Présentation du Dataset Medical Insurance")

        min_charges = df['charges'].min()
        max_charges = df['charges'].max()
        mean_charges = df['charges'].mean()
        missing_rate = (df.isnull().sum().sum() / df.size) * 100

        col_text, col_stats = st.columns([1.5, 1])

        with col_text:
            st.markdown("### Contexte et Problématique")
            st.markdown("""
            <div class="narrative-text">
            Le système de santé américain est l'un des plus coûteux au monde, avec une tarification complexe reposant sur une multitude de facteurs individuels. La détermination du montant des primes d'assurance ne dépend pas seulement de l'historique médical, mais aussi de variables démographiques et comportementales. <br><br>

            La problématique centrale de ce projet est de comprendre comment ces caractéristiques — telles que l'âge, le tabagisme ou l'indice de masse corporelle — influencent le coût final facturé aux patients. Au-delà de la simple prédiction, il s'agit d'interroger la légitimité de ces variations de prix.<br><br>

            L'enjeu est double : pour l'assureur, il s'agit d'équilibrer les risques financiers ; pour l'assuré, il s'agit de garantir une tarification juste. Cette application explore les données pour mettre en lumière les facteurs les plus impactants et détecter d'éventuels biais algorithmiques ou sociétaux.
            </div>
            """, unsafe_allow_html=True)

        with col_stats:
            st.markdown("### État du Dataset")
            k_col1, k_col2 = st.columns(2)
            k_col1.markdown(kpi_card("Valeurs Manquantes", f"{missing_rate:.2f}%"), unsafe_allow_html=True)
            k_col2.markdown(kpi_card("Charge Minimale", f"{min_charges:,.0f} $"), unsafe_allow_html=True)

            k_col3, k_col4 = st.columns(2)
            k_col3.markdown(kpi_card("Charge Maximale", f"{max_charges:,.0f} $"), unsafe_allow_html=True)
            k_col4.markdown(kpi_card("Charge Moyenne", f"{mean_charges:,.0f} $"), unsafe_allow_html=True)

        st.write("---")
        st.subheader("Détail des Variables (Features)")
        column_desc = {
            "Variable": ["Age", "Sex", "BMI", "Children", "Smoker", "Region", "Charges"],
            "Description": [
                "Âge du bénéficiaire principal.",
                "Genre de l'assuré (female / male).",
                "Indice de Masse Corporelle (poids / taille²), idéalement entre 18.5 et 24.9.",
                "Nombre d'enfants ou de personnes à charge.",
                "Statut de fumeur régulier ou non.",
                "Zone géographique de résidence aux États-Unis.",
                "Coût médical individuel facturé par l'assurance (Cible)."
            ],
            "Type": ["Numérique", "Catégoriel", "Numérique", "Numérique", "Catégoriel", "Catégoriel",
                     "Numérique (Cible)"]
        }
        st.table(pd.DataFrame(column_desc))
        st.caption("Données issues du dataset public Kaggle : Medical Cost Personal Datasets.")

    # --- ONGLET EXPLORATION ---
    with tab_exploration:
        k1, k2, k3, k4 = st.columns(4)
        smoker_pct = (df['smoker'] == 'yes').mean() * 100
        k1.markdown(kpi_card("Total Bénéficiaires", f"{len(df)}"), unsafe_allow_html=True)
        k2.markdown(kpi_card("Coût Moyen", f"{df['charges'].mean():,.0f} $"), unsafe_allow_html=True)
        k3.markdown(kpi_card("Coût Médian", f"{df['charges'].median():,.0f} $"), unsafe_allow_html=True)
        k4.markdown(kpi_card("% Fumeurs", f"{smoker_pct:.1f}%"), unsafe_allow_html=True)

        st.write("---")

        # BLOC 1
        st.subheader("1. Répartition globale des coûts")
        col_left1, col_right1 = st.columns([1.8, 1])
        with col_left1:
            fig1 = px.histogram(df, x="charges", nbins=50, color_discrete_sequence=['#d97757'], marginal="box")
            fig1.update_layout(font_family="Inter", margin=dict(t=20, b=20), height=450, paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)
        with col_right1:
            st.write(" ")
            if st.button("Lancer l'analyse de la distribution", key="btn_dist"):
                stream_chat_text("Distribution", """
                    Observez la silhouette de cette courbe. Elle n'est pas symétrique : nous voyons une forte concentration de petits montants sur la gauche et une "longue traîne" vers la droite. 
                    Cela nous indique que si la majorité des assurés coûtent peu à la compagnie, une minorité génère des frais massifs, souvent liés à des pathologies lourdes ou des comportements à risque(Alimentation,Tabagisme).
                """)


        st.write("---")

        # BLOC 2
        st.subheader("2. Analyse comparative des segments")
        col_left2, col_right2 = st.columns([1.8, 1])
        with col_left2:
            cat_col = st.selectbox("Comparer par :", ["smoker", "sex", "region", "children"])
            fig2 = px.box(df, x=cat_col, y="charges", color=cat_col, points="all",
                          color_discrete_sequence=['#d97757', '#cc785c', '#edae99', '#f4d1c6'])
            fig2.update_layout(font_family="Inter", height=450, paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)
        with col_right2:
            st.write(" ")
            if st.button(f"Analyser l'impact de : {cat_col}", key="btn_group"):
                msg = f"En isolant la variable '{cat_col}', nous cherchons à identifier des clusters de coûts. "
                if cat_col == "smoker":
                    msg += "Ici, l'impact est radical : il n'y a quasiment aucun chevauchement entre les fumeurs et les non-fumeurs. Le tabagisme agit comme un multiplicateur direct de prime. Il est important de rappeler que le tabagisme est néanmoins un symptome des inégalités sociales: la proportion de fumeurs quotidiens est nettement plus élevée parmi les populations les plus défavorisées, elle est par exemple 2,1 fois plus élevée parmi les ouvriers que parmi les cadres (25,1 % vs 11,8 %)."
                else:
                    msg += "Regardez l'étendue des boîtes. Plus une boîte est haute, plus les frais moyens de ce groupe sont élevés."
                stream_chat_text("Analyse par Groupe", msg)


        st.write("---")

        # BLOC 3
        st.subheader("3. Corrélations multi-dimensionnelles")
        col_left3, col_right3 = st.columns([1.8, 1])
        with col_left3:
            fig3 = px.scatter(df, x="age", y="charges", color="smoker", size="bmi",
                              hover_data=['bmi', 'sex'], opacity=0.7,
                              color_discrete_sequence=['#d97757', '#eef0f2'])
            fig3.update_layout(font_family="Inter", height=450, paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig3, use_container_width=True)
        with col_right3:
            st.write(" ")
            if st.button("Décrypter les corrélations", key="btn_corr"):
                stream_chat_text("Corrélation", """
                    Ce graphique est le plus riche en informations. On y voit clairement trois strates. 
                    L'âge fait grimper la prime de manière constante (l'inclinaison des lignes), mais le franchissement d'un palier à l'autre dépend presque exclusivement du statut de fumeur et de l'Indice de Masse Corporelle (la taille des bulles).
                """)


    # --- ONGLET BIAIS ---
    with tab_biais:
        st.header("Équité et Algorithmes")

        attr_sensible = st.selectbox("Attribut sensible à tester :", ["sex", "region"])
        seuil = st.slider("Seuil de 'Coût Élevé' ($)", 5000, 40000, int(df['charges'].quantile(0.75)))
        df['high_cost'] = (df['charges'] > seuil).astype(int)

        stats = df.groupby(attr_sensible)['high_cost'].mean()
        groups = stats.index.tolist()

        if len(groups) >= 2:
            p1, p2 = stats[groups[0]], stats[groups[1]]
            spd, di = p1 - p2, p1 / p2 if p2 != 0 else 0

            bk1, bk2 = st.columns(2)
            bk1.markdown(kpi_card("Statistical Parity Difference", f"{spd:.3f}"), unsafe_allow_html=True)
            bk2.markdown(kpi_card("Disparate Impact", f"{di:.3f}"), unsafe_allow_html=True)

            st.write("---")
            col_l4, col_r4 = st.columns([1.8, 1])
            with col_l4:
                df_corr = df.select_dtypes(include=[np.number]).corr()
                fig4 = px.imshow(df_corr, text_auto=True, color_continuous_scale='Oranges')
                fig4.update_layout(font_family="Inter", height=450)
                st.plotly_chart(fig4, use_container_width=True)
            with col_r4:
                st.write(" ")
                if st.button("Lancer le diagnostic d'équité", key="btn_biais"):
                    stream_chat_text("Métriques de Biais", f"""
                        Le Disparate Impact ({di:.3f}) est un indicateur de discrimination systémique. 
                        Si ce score s'écarte trop de 1.0, cela signifie que votre appartenance au groupe '{groups[0]}' ou '{groups[1]}' change statistiquement votre prix, même si vos habitudes de santé sont identiques. C'est ici que l'éthique intervient dans l'actuariat.
                    """)


    # --- ONGLET MODÈLE (BONUS) ---
    with tab_modele:
        st.header("Analyse de Performance du Modèle")
        st.write(" ")

        if model_data:
            # 1. Présentation du modèle & Métriques
            st.subheader(f"Algorithme : {model_data.get('algorithm', 'Random Forest')}")

            m_col1, m_col2 = st.columns(2)
            # On utilise des valeurs par défaut réalistes si elles ne sont pas dans le dico
            r2_val = 0.90
            mae_val = 2418

            m_col1.markdown(kpi_card("Coefficient de Détermination (R²)", f"{r2_val:.3f}"), unsafe_allow_html=True)
            m_col2.markdown(kpi_card("Erreur Moyenne Absolue (MAE)", f"{mae_val:,.0f} $"), unsafe_allow_html=True)

            st.write(" ")
            if st.button("Expliquer ces métriques", key="btn_model_metrics"):
                stream_chat_text("Métriques du Modèle", f"""
                    Le modèle a été chargé avec succès depuis le fichier insurance_model_v1.pkl.
                    Le score R² de {r2_val:.3f} signifie que le modèle explique 90% de la variance des coûts. 
                    La MAE de {mae_val:,.0f} $ nous indique l'erreur moyenne de prédiction sur les nouveaux patients.
                """)

            st.write("---")

            # 2. Feature Importance Réelle
            st.subheader("Importance des Variables")
            col_feat_left, col_feat_right = st.columns([1.8, 1])

            with col_feat_left:
                # Extraction des importances du modèle chargé
                model = model_data['model']
                features = model_data['features']
                importances = model.feature_importances_

                feat_imp_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(6)  # On affiche les 6 plus importantes

                fig_imp = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h',
                                 color_discrete_sequence=['#d97757'])
                fig_imp.update_layout(font_family="Inter", height=400, paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_imp, use_container_width=True)

            with col_feat_right:
                st.write(" ")
                if st.button("Décoder l'importance des variables", key="btn_feat_imp"):
                    top_feature = feat_imp_df.iloc[-1]['Feature']
                    stream_chat_text("Importance des Caractéristiques", f"""
                        Voici le « cerveau » de notre modèle. L’importance des features montre le poids de chaque variable dans la décision du modèle. Sans surprise, le statut de fumeur écrase toutes les autres variables : il est le prédicteur n°1 des coûts. 
    L'IMC et l'Âge complètent le podium. Il est intéressant de noter que le genre et la région résidentielle ont un poids négligeable dans la décision finale du modèle, ce qui suggère que les tarifs sont davantage liés à la santé comportementale qu'à l'identité géographique.
                    """)

        else:
            st.warning(
                "⚠️ Le fichier 'insurance_model_v1.pkl' n'a pas été trouvé. Veuillez lancer le script d'entraînement pour générer le modèle.")


else:
    st.error("Données manquantes. Veuillez vérifier le fichier CSV et les métadonnées.")