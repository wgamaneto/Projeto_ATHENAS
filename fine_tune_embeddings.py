import json
import os
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def main():
    dataset_path = Path("golden_dataset.json")
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    train_examples = [
        InputExample(texts=[item["pergunta"], item["resposta_ideal"]]) for item in data
    ]

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=0,
        show_progress_bar=True,
    )

    output_path = Path(os.getenv("LOCAL_EMBEDDING_MODEL", "local_embedding_model"))
    model.save(str(output_path))
    print(f"Modelo salvo em {output_path.resolve()}")


if __name__ == "__main__":
    main()
