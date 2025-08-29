import re
from collections import Counter
from typing import Tuple, Counter as CounterType

LOG_FILE = "feedback.log"


def read_feedback_log() -> Tuple[int, int, CounterType[str]]:
    """Lê o arquivo de feedback e retorna contagem de positivos, negativos e perguntas negativas."""
    positivos = 0
    negativos = 0
    perguntas_negativas: CounterType[str] = Counter()
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for linha in f:
                if "Feedback positivo" in linha:
                    positivos += 1
                elif "Feedback negativo" in linha:
                    negativos += 1
                    match = re.search(r"Feedback negativo para a pergunta: (.*)", linha)
                    if match:
                        pergunta = match.group(1).strip()
                        perguntas_negativas[pergunta] += 1
    except FileNotFoundError:
        pass
    return positivos, negativos, perguntas_negativas


def summarize_feedback() -> None:
    """Imprime um resumo do feedback coletado."""
    positivos, negativos, perguntas_negativas = read_feedback_log()
    total = positivos + negativos
    pct_pos = (positivos / total * 100) if total else 0
    pct_neg = (negativos / total * 100) if total else 0

    print(f"Feedbacks positivos: {positivos} ({pct_pos:.2f}%)")
    print(f"Feedbacks negativos: {negativos} ({pct_neg:.2f}%)")

    if perguntas_negativas:
        print("\nPerguntas com mais feedbacks negativos:")
        for pergunta, contagem in perguntas_negativas.most_common(5):
            print(f"- {pergunta}: {contagem}")


if __name__ == "__main__":
    summarize_feedback()
