import json
import logging

import requests
import streamlit as st

st.set_page_config(page_title="Assistente de Conhecimento ATHENAS")

st.title("Assistente de Conhecimento ATHENAS")

logging.basicConfig(
    filename="feedback.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

if "historico" not in st.session_state:
    st.session_state["historico"] = []

for i, mensagem in enumerate(st.session_state["historico"]):
    st.markdown(f"**Você:** {mensagem['pergunta']}")
    st.markdown(f"**ATHENAS:** {mensagem['resposta']}")

    col_up, col_down = st.columns(2)
    if col_up.button("👍", key=f"up_{i}"):
        logging.info("Feedback positivo para a pergunta: %s", mensagem["pergunta"])
    if col_down.button("👎", key=f"down_{i}"):
        logging.info(
            "Feedback negativo para a pergunta: %s | Fontes: %s",
            mensagem["pergunta"],
            json.dumps(mensagem.get("fontes", []), ensure_ascii=False),
        )

    fontes_previas = mensagem.get("fontes", [])
    if fontes_previas:
        with st.expander("Fontes utilizadas"):
            for fonte in fontes_previas:
                origem = fonte.get("fonte", "")
                texto = fonte.get("texto", "")
                st.markdown(f"**Fonte:** {origem}")
                st.write(texto)

pergunta = st.text_input("Digite sua pergunta:", key="pergunta_input")

if st.button("Enviar"):
    pergunta = st.session_state.get("pergunta_input", "")
    if pergunta:
        try:
            with st.spinner("ATHENAS está pensando..."):
                historico = st.session_state["historico"][-5:]
                resp = requests.get(
                    "http://localhost:8000/answer",
                    params={
                        "pergunta": pergunta,
                        "historico": json.dumps(historico, ensure_ascii=False),
                    },
                )
            if resp.ok:
                data = resp.json()
                resposta = data.get("resposta", "")
                fontes = data.get("fontes", [])
                st.session_state["historico"].append(
                    {"pergunta": pergunta, "resposta": resposta, "fontes": fontes}
                )
                st.session_state["pergunta_input"] = ""
            else:
                st.error("Desculpe, algo deu errado. Tente novamente em alguns instantes.")
        except Exception as e:
            logging.error("Erro ao conectar ao backend: %s", e)
            st.error(
                "Desculpe, não consegui me conectar à ATHENAS. Tente novamente em alguns instantes."
            )
    else:
        st.warning("Digite uma pergunta antes de enviar.")


st.header("Gerar apresentação")
topico = st.text_input("Tópico da apresentação", key="topico_apresentacao")
num_slides = st.number_input(
    "Número de slides", min_value=1, max_value=20, value=5, key="num_slides_apresentacao"
)

if st.button("Gerar apresentação"):
    if topico:
        try:
            with st.spinner("Gerando apresentação..."):
                resp = requests.get(
                    "http://localhost:8000/gerar_apresentacao",
                    params={"topico": topico, "num_slides": int(num_slides)},
                )
            if resp.ok:
                data = resp.json()
                slides = data.get("slides", [])
                for i, slide in enumerate(slides, 1):
                    st.markdown(f"**Slide {i}:** {slide}")
            else:
                st.error("Desculpe, não foi possível gerar a apresentação.")
        except Exception as e:
            logging.error("Erro ao gerar apresentação: %s", e)
            st.error("Não consegui conectar ao serviço de geração.")
    else:
        st.warning("Digite um tópico antes de gerar.")
