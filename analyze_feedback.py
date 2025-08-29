import json
import re
from collections import Counter
from typing import Dict, List, Tuple, Counter as CounterType

LOG_FILE = "feedback.log"
ERROR_FILE = "erros_para_analise.json"


def read_feedback_log() -> Tuple[int, int, CounterType[str], List[Dict[str, List[Dict[str, str]]]]]:
    """Lê o arquivo de feedback e retorna estatísticas e detalhes de erros.

    Além das contagens de feedbacks positivos e negativos, também coleta as
    perguntas com feedback negativo e os respectivos *chunks* de contexto
    utilizados na resposta para gerar um relatório de erros.
    """

    positivos = 0
    negativos = 0
    perguntas_negativas: CounterType[str] = Counter()
    erros: List[Dict[str, List[Dict[str, str]]]] = []

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for linha in f:
                if "Feedback positivo" in linha:
                    positivos += 1
                elif "Feedback negativo" in linha:
                    negativos += 1
                    match = re.search(
                        r"Feedback negativo para a pergunta: (.*?)(?:\s\|\sFontes: (.*))?$",
                        linha,
                    )
                    if match:
                        pergunta = match.group(1).strip()
                        perguntas_negativas[pergunta] += 1
                        fontes_raw = match.group(2)
                        fontes = []
                        if fontes_raw:
                            try:
                                fontes = json.loads(fontes_raw)
                            except json.JSONDecodeError:
                                pass
                        erros.append({"pergunta": pergunta, "fontes": fontes})
    except FileNotFoundError:
        pass

    return positivos, negativos, perguntas_negativas, erros


def save_errors(erros: List[Dict[str, List[Dict[str, str]]]]) -> None:
    """Salva as perguntas com feedback negativo e seus contextos em JSON."""

    with open(ERROR_FILE, "w", encoding="utf-8") as f:
        json.dump(erros, f, ensure_ascii=False, indent=2)


def summarize_feedback() -> None:
    """Imprime um resumo do feedback coletado e gera relatório de erros."""

    positivos, negativos, perguntas_negativas, erros = read_feedback_log()
    total = positivos + negativos
    pct_pos = (positivos / total * 100) if total else 0
    pct_neg = (negativos / total * 100) if total else 0

    print(f"Feedbacks positivos: {positivos} ({pct_pos:.2f}%)")
    print(f"Feedbacks negativos: {negativos} ({pct_neg:.2f}%)")

    if perguntas_negativas:
        print("\nPerguntas com mais feedbacks negativos:")
        for pergunta, contagem in perguntas_negativas.most_common(5):
            print(f"- {pergunta}: {contagem}")

    save_errors(erros)
    print(f"\nArquivo '{ERROR_FILE}' gerado com {len(erros)} itens para análise.")


if __name__ == "__main__":
    summarize_feedback()
