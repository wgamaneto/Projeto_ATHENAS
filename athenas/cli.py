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

    # Documentos de exemplo. Em versões futuras serão substituídos por dados corporativos.
    documentos = [
        "A ATHENAS é um assistente IA conversacional que unifica o conhecimento interno.",
        "O objetivo inicial é reduzir o tempo médio gasto na busca por informações.",
    ]

    rag = AthenasRAG()
    resposta = rag.answer(args.pergunta, documentos)
    print(resposta)


if __name__ == "__main__":
    main()
