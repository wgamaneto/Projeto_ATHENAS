# Projeto ATHENAS

Protótipo inicial da plataforma de **Inteligência de Acesso ao Conhecimento**. Este repositório contém uma implementação mínima do fluxo de *Retrieval-Augmented Generation* (RAG) utilizado pela ATHENAS e serve como ponto de partida para a evolução do MVP.

## Estrutura do Repositório

- `athenas/core.py` – pipeline básico de RAG com componentes substituíveis e
  roteamento simples entre modelos da OpenAI.
- `athenas/cli.py` – interface de linha de comando de demonstração.
- `ingest.py` – script para ingestão e anonimização de documentos no ChromaDB.
- `backend/` – API FastAPI que expõe o pipeline de perguntas e respostas.
- `knowledge_base/` – documentos de exemplo ingeridos no banco vetorial.
- `docker-compose.yml` – orquestração do backend e do ChromaDB para desenvolvimento.

## Pré-requisitos

- Python 3.10+
- [OpenAI API key](https://platform.openai.com/) disponível na variável de ambiente `OPENAI_API_KEY`
- Docker e Docker Compose (opcional, para execução com contêineres)

Crie um arquivo `.env` na raiz com as variáveis necessárias:

```bash
OPENAI_API_KEY="sua-chave"
OPENAI_CHAT_MODEL="gpt-4o-mini"
OPENAI_FAST_MODEL="gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL="text-embedding-ada-002"
CROSS_ENCODER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
# opcional
CHROMA_HOST="localhost"
CHROMA_PORT="8001"
```

O pipeline seleciona automaticamente entre os modelos definidos em
`OPENAI_CHAT_MODEL` e `OPENAI_FAST_MODEL` de acordo com a complexidade da
pergunta, equilibrando custo e desempenho.

## Instalação

```bash
git clone <repo>
cd Projeto_ATHENAS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução do CLI

Execute uma pergunta diretamente pelo terminal:

```bash
python -m athenas.cli "Qual a proposta da ATHENAS?"
```

## Ingestão de Documentos

Para popular o banco vetorial com os arquivos de `knowledge_base/`:

```bash
python ingest.py
```

O script divide os documentos em *chunks*, anonimiza informações sensíveis com o Presidio e persiste embeddings no ChromaDB.

## API com Docker

Uma API de demonstração pode ser levantada com Docker:

```bash
docker-compose up --build
```

Após a inicialização o endpoint estará disponível em `http://localhost:8000/answer`. 
O parâmetro opcional `grupos` permite informar as permissões do usuário. Exemplo:

```
http://localhost:8000/answer?pergunta=Qual%20o%20objetivo%20do%20projeto?&grupos=vendas,diretoria
```

O serviço `vector-db` expõe a porta `8001` apenas para fins de demonstração.

## Contribuição

Sinta-se à vontade para abrir *issues* e *pull requests* com melhorias ou correções.
