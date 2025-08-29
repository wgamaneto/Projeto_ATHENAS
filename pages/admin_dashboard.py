import pandas as pd
import streamlit as st
from analyze_feedback import read_feedback_log

st.set_page_config(page_title="Dashboard de Feedback")

st.title("Dashboard de Feedback")

positivos, negativos, perguntas_negativas = read_feedback_log()
total = positivos + negativos

col1, col2 = st.columns(2)
col1.metric("Positivos", positivos)
col2.metric("Negativos", negativos)

if total:
    st.write(f"Porcentagem de feedbacks positivos: {positivos / total * 100:.2f}%")
    st.write(f"Porcentagem de feedbacks negativos: {negativos / total * 100:.2f}%")
else:
    st.info("Nenhum feedback registrado ainda.")

if perguntas_negativas:
    st.subheader("Perguntas com mais feedbacks negativos")
    df = pd.DataFrame(perguntas_negativas.most_common(), columns=["Pergunta", "Negativos"])
    st.table(df)
