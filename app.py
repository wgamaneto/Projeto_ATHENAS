import streamlit as st
import requests

st.set_page_config(page_title="Assistente de Conhecimento ATHENAS")

st.title("Assistente de Conhecimento ATHENAS")

pergunta = st.text_input("Digite sua pergunta:")

resposta_area = st.empty()

if st.button("Enviar"):
    if pergunta:
        try:
            resp = requests.get("http://localhost:8000/answer", params={"pergunta": pergunta})
            if resp.ok:
                data = resp.json()
                resposta_area.write(data.get("resposta", ""))
            else:
                resposta_area.error(f"Erro {resp.status_code}")
        except Exception as e:
            resposta_area.error(f"Erro ao conectar ao backend: {e}")
    else:
        resposta_area.warning("Digite uma pergunta antes de enviar.")
