# Projeto ATHENAS

Protótipo inicial da plataforma de **Inteligência de Acesso ao Conhecimento**
descrita no documento PDF deste repositório.

Este repositório contém uma implementação mínima do fluxo RAG (Retrieval-Augmented Generation)
utilizado pela ATHENAS. O objetivo é servir como ponto de partida para evolução do MVP.

## Estrutura

- `athenas/core.py` – pipeline básico de RAG com componentes substituíveis.
- `athenas/cli.py` – interface de linha de comando de demonstração.

## Como executar

```bash
python -m athenas.cli "Qual a proposta da ATHENAS?"
```

O comando utiliza documentos fictícios apenas para ilustrar o fluxo.
