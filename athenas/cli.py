"""Interface de linha de comando para o protótipo da ATHENAS."""

import argparse

from .core import AthenasRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Protótipo inicial da ATHENAS")
    parser.add_argument(
        "pergunta",
        nargs="?",
        default="Qual o objetivo do projeto?",
        help="Pergunta a ser respondida.",
    )
    args = parser.parse_args()

    rag = AthenasRAG()
    resposta = rag.answer(args.pergunta)
    print({"pergunta": args.pergunta, "resposta": resposta})


if __name__ == "__main__":
    main()
