import streamlit as st
import requests

st.set_page_config(page_title="Assistente de Conhecimento ATHENAS")

st.title("Assistente de Conhecimento ATHENAS")

if "historico" not in st.session_state:
    st.session_state["historico"] = []

for mensagem in st.session_state["historico"]:
    st.markdown(f"**Você:** {mensagem['pergunta']}")
    st.markdown(f"**ATHENAS:** {mensagem['resposta']}")
    fontes_previas = mensagem.get("fontes", [])
    if fontes_previas:
        with st.expander("Fontes utilizadas"):
            for fonte in fontes_previas:
                st.write(fonte)

pergunta = st.text_input("Digite sua pergunta:", key="pergunta_input")

if st.button("Enviar"):
    pergunta = st.session_state.get("pergunta_input", "")
    if pergunta:
        try:
            resp = requests.get("http://localhost:8000/answer", params={"pergunta": pergunta})
            if resp.ok:
                data = resp.json()
                resposta = data.get("resposta", "")
                fontes = data.get("fontes", [])
                st.session_state["historico"].append(
                    {"pergunta": pergunta, "resposta": resposta, "fontes": fontes}
                )
                st.session_state["pergunta_input"] = ""
            else:
                st.error(f"Erro {resp.status_code}")
        except Exception as e:
            st.error(f"Erro ao conectar ao backend: {e}")
    else:
        st.warning("Digite uma pergunta antes de enviar.")
