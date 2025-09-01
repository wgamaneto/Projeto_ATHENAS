"""Avaliação de respostas usando LLM como juiz.

Este script executa o pipeline da ATHENAS em um conjunto de perguntas
(`golden_dataset.json`) e utiliza um modelo de linguagem para avaliar a
qualidade das respostas. O avaliador atribui notas de 1 a 5 para os
critérios de correção, completude e relevância, bem como uma nota geral.
"""

from __future__ import annotations

import json
import os
from statistics import mean
from typing import Any, Dict

from athenas.core import AthenasRAG


def _llm_judge(
    pergunta: str,
    resposta_obtida: str,
    resposta_ideal: str,
    *,
    model: str | None = None,
) -> Dict[str, Any]:
    """Avalia a resposta usando um LLM.

    Retorna um dicionário com as notas e justificativas para os critérios
    de correção, completude e relevância, além de uma ``nota_geral``.
    """

    from openai import OpenAI

    client = OpenAI()
    model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    system_prompt = (
        "Você é um avaliador de respostas de sistemas de QA. Para cada"
        " pergunta, compare a resposta obtida com a resposta ideal e"
        " atribua notas de 1 a 5 para os critérios de correção, completude"
        " e relevância. Forneça justificativas curtas para cada critério e"
        " calcule também uma nota_geral. Responda APENAS em JSON no formato:\n"
        "{\n  'correcao': {'nota': <int>, 'justificativa': <str>},\n"
        "  'completude': {'nota': <int>, 'justificativa': <str>},\n"
        "  'relevancia': {'nota': <int>, 'justificativa': <str>},\n"
        "  'nota_geral': <int>\n}"
    )

    user_prompt = (
        f"Pergunta: {pergunta}\n\n"
        f"Resposta ideal: {resposta_ideal}\n\n"
        f"Resposta obtida: {resposta_obtida}"
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(completion.choices[0].message.content)


def evaluate(dataset_path: str = "golden_dataset.json") -> None:
    """Avalia o desempenho da ATHENAS em um conjunto de perguntas."""

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    rag = AthenasRAG()

    notas_gerais = []
    for item in dataset:
        pergunta = item["pergunta"]
        resposta_ideal = item["resposta_ideal"]

        try:
            resposta_obtida, _, _ = rag.answer(pergunta)
        except Exception as exc:  # pragma: no cover - falha no pipeline
            resposta_obtida = f"Erro ao obter resposta: {exc}"

        try:
            avaliacao = _llm_judge(pergunta, resposta_obtida, resposta_ideal)
            nota = avaliacao.get("nota_geral")
            if isinstance(nota, (int, float)):
                notas_gerais.append(nota)
        except Exception as exc:  # pragma: no cover - falha na API
            avaliacao = {"erro": str(exc)}

        print(f"Pergunta: {pergunta}")
        print(f"Resposta obtida: {resposta_obtida}")
        print(f"Resposta ideal: {resposta_ideal}")
        print(f"Avaliação: {json.dumps(avaliacao, ensure_ascii=False)}\n")

    if notas_gerais:
        print(f"Nota média geral: {mean(notas_gerais):.2f}")
    else:
        print("Não foi possível calcular a nota média geral.")


if __name__ == "__main__":
    evaluate()

