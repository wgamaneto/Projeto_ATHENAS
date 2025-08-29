import json
from difflib import SequenceMatcher
from athenas.core import AthenasRAG


def evaluate(dataset_path: str = "golden_dataset.json", threshold: float = 0.6) -> None:
    """Avalia o desempenho da ATHENAS em um conjunto de perguntas e respostas."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    rag = AthenasRAG()

    acertos = 0
    for item in dataset:
        pergunta = item["pergunta"]
        resposta_ideal = item["resposta_ideal"]

        try:
            resposta_obtida, _ = rag.answer(pergunta)
        except Exception as exc:
            resposta_obtida = f"Erro: {exc}"

        similaridade = SequenceMatcher(None, resposta_obtida.lower(), resposta_ideal.lower()).ratio()
        correto = similaridade >= threshold
        acertos += int(correto)

        print(f"Pergunta: {pergunta}")
        print(f"Resposta obtida: {resposta_obtida}")
        print(f"Resposta ideal: {resposta_ideal}")
        print(f"Similaridade: {similaridade:.2f} -> {'OK' if correto else 'FALHOU'}\n")

    total = len(dataset)
    print(f"Acurácia total: {acertos / total:.2%} ({acertos}/{total})")


if __name__ == "__main__":
    evaluate()
