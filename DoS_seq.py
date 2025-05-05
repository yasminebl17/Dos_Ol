import streamlit as st
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier

# 🌙 Thème sombre customisé
st.set_page_config(page_title="Détection d'Intrusion", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #1e1e1e;
        color: #00ffcc;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔐 Détection séquentielle des attaques en temps réel")
st.markdown("---")

uploaded_file = st.file_uploader("📂 Choisir un fichier CSV (ex: KDDTest-21.csv)", type="csv")

if uploaded_file:
    with st.spinner("Chargement et préparation des données..."):
        data = pd.read_csv(uploaded_file)
        data.dropna(inplace=True)

        label_encoders = {}
        for col in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

        attack_label_value = None
        target_column = data.columns[-1]
        if "DoS" in label_encoders[target_column].classes_:
            attack_label_value = label_encoders[target_column].transform(["DoS"])[0]
            st.info(f"La classe 'Dos' est encodée en : {attack_label_value}")
        else:
            st.warning("❗ La classe 'Dos' n'existe pas dans ce dataset.")

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        st.success("✅ Données prêtes. Tu peux lancer la simulation maintenant !")

        if st.button("🚀 Lancer la Simulation"):
            if attack_label_value is None:
                st.error("La simulation ne peut pas démarrer : la classe 'Dos' n'a pas été trouvée.")
            else:
                with st.spinner("Simulation séquentielle en cours..."):
                    ATTACK_CLASSES = [attack_label_value]
                    true_labels = []
                    predicted_labels = []
                    historique = pd.DataFrame(columns=data.columns)

                    def get_batch():
                        return data.sample(n=700).reset_index(drop=True)

                    def simulate_slave(batch_data):
                        X_new = scaler.transform(batch_data.iloc[:, :-1])
                        predictions = knn.predict(X_new)
                        batch_data["Prediction"] = predictions
                        nb_attacks = sum(pred in ATTACK_CLASSES for pred in predictions)
                        return batch_data, nb_attacks, predictions, batch_data.iloc[:, -2].tolist()

                    start_time = time.time()
                    total_alerts = 0

                    for i in range(10):
                        st.markdown(f"### 📦 Lot de données {i+1}")
                        batch_data = get_batch()

                        # Exécution SÉQUENTIELLE
                        for j in range(3):
                            batch_result, nb_attacks, preds, trues = simulate_slave(batch_data.copy())
                            predicted_labels.extend(preds)
                            true_labels.extend(trues)
                            historique = pd.concat([historique, batch_result], ignore_index=True)
                            total_alerts += nb_attacks
                            st.markdown(f"⚠️ Attaques détectées dans le sous-lot {j+1} : **{nb_attacks}**")

                        time.sleep(1)

                    end_time = time.time()
                    execution_time = end_time - start_time
                    total_flux = len(true_labels)
                    nb_attacks_total = sum(label in ATTACK_CLASSES for label in true_labels)
                    nb_normaux = total_flux - nb_attacks_total
                    attack_percentage = (nb_attacks_total / total_flux) * 100 if total_flux > 0 else 0
                    precision = precision_score(true_labels, predicted_labels, average='macro')

                    st.success("✅ Simulation terminée avec succès.")
                    st.markdown("### 📊 Statistiques globales :")
                    st.markdown(f"- Total de flux traités : **{total_flux}**")
                    st.markdown(f"- Total de flux d’attaque : **{nb_attacks_total}**")
                    st.markdown(f"- Flux normaux : **{nb_normaux}**")
                    st.markdown(f"- Pourcentage d’attaques : **{attack_percentage:.2f}%**")
                    st.markdown(f"- Précision globale : **{precision:.4f}**")
                    st.markdown(f"- Temps d'exécution : **{execution_time:.2f} secondes**")

                    # Alertes visuelles
                    if attack_percentage > 70:
                        st.markdown(
                            f"<div style='background-color:#ff0000;padding:20px;border-radius:10px;'>"
                            f"<h2 style='color:white;text-align:center;'>🚨 DANGER CRITIQUE : {attack_percentage:.2f}% d’attaques !</h2>"
                            f"<h4 style='text-align:center;color:white;'>⚠️ Situation extrêmement grave. ⚠️</h4></div>",
                            unsafe_allow_html=True
                        )
                    elif attack_percentage > 50:
                        st.markdown(
                            f"<div style='background-color:#ff9900;padding:20px;border-radius:10px;'>"
                            f"<h2 style='color:white;text-align:center;'>⚠️ DANGER : {attack_percentage:.2f}% d’attaques !</h2>"
                            f"<h4 style='text-align:center;color:white;'>⚠️ Situation préoccupante. ⚠️</h4></div>",
                            unsafe_allow_html=True
                        )
                    elif attack_percentage > 20:
                        st.markdown(
                            f"<div style='background-color:#ffff66;padding:20px;border-radius:10px;'>"
                            f"<h2 style='color:#333;text-align:center;'>⚠️ Attention : {attack_percentage:.2f}% d’attaques.</h2>"
                            f"<h4 style='text-align:center;color:#333;'>⚠️ À surveiller de près. ⚠️</h4></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div style='background-color:#00cc66;padding:20px;border-radius:10px;'>"
                            f"<h2 style='color:white;text-align:center;'>✅ Système Stable : {attack_percentage:.2f}% d’attaques détectées.</h2>"
                            f"<h4 style='text-align:center;color:white;'>👍 Situation normale.</h4></div>",
                            unsafe_allow_html=True
                        )
