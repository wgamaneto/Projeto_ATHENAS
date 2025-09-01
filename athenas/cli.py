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
    resultado = rag.executar(args.pergunta)
    resultado["pergunta"] = args.pergunta
    print(resultado)


if __name__ == "__main__":
    main()
