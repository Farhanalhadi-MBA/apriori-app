
import pandas as pd
from apyori import apriori
import streamlit as st

st.set_page_config(page_title="Apriori Web App", layout="wide")

st.title("🛒 Apriori Market Basket Analysis (Web App)")

uploaded_file = st.file_uploader("Upload Market_Basket_Optimisation.csv", type="csv")

if uploaded_file:
    try:
        dataset = pd.read_csv(uploaded_file, header=None)
        transactions = [
            [str(dataset.values[i, j]) for j in range(20) if str(dataset.values[i, j]) != 'nan']
            for i in range(len(dataset))
        ]

        rules = apriori(
            transactions=transactions,
            min_support=0.003,
            min_confidence=0.2,
            min_lift=3,
            min_length=2,
            max_length=2
        )

        results = list(rules)

        def inspect(results):
            lhs = [tuple(result.ordered_statistics[0].items_base)[0] for result in results]
            rhs = [tuple(result.ordered_statistics[0].items_add)[0] for result in results]
            support = [result.support for result in results]
            confidence = [result.ordered_statistics[0].confidence for result in results]
            lift = [result.ordered_statistics[0].lift for result in results]
            return list(zip(lhs, rhs, support, confidence, lift))

        df = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
        df = df.nlargest(10, 'Lift')

        st.success("Analisis selesai! Berikut hasil 10 aturan terbaik:")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Hasil CSV", csv, "apriori_output.csv", "text/csv")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
else:
    st.info("Silakan upload file CSV untuk memulai.")
