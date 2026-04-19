#!/bin/bash

mlflow server --port 5000

export MLFLOW_TRACKING_URI=http://localhost:5000

mlflow run -e ner -P config_file=configs/conll2003-ai.toml .
mlflow run -e ner -P config_file=configs/conll2003-literature.toml .
mlflow run -e ner -P config_file=configs/conll2003-music.toml .
mlflow run -e ner -P config_file=configs/conll2003-politics.toml .
mlflow run -e ner -P config_file=configs/conll2003-science.toml .


mlflow run -e ner -P config_file=configs/conll2003-ai-pile.toml .
mlflow run -e ner -P config_file=configs/conll2003-literature-pile.toml .
mlflow run -e ner -P config_file=configs/conll2003-music-pile.toml .
mlflow run -e ner -P config_file=configs/conll2003-politics-pile.toml .
mlflow run -e ner -P config_file=configs/conll2003-science-pile.toml .



python -m owner.visualize_entity_embeddings \
  --checkpoint checkpoints/owner/conll2003/new_model/entity_typing.pt \
  --output outputs/crossner_entity_tsne_soft_tuning.png



  python -m owner.visualize_entity_embeddings \
  --checkpoint checkpoints/owner/conll2003/100/entity_typing.pt \
  --output outputs/crossner_entity_tsne_wt_er.png


python -m owner.visualize_entity_embeddings \
  --checkpoint checkpoints/owner/conll2003/100/entity_typing.pt \
  --output outputs/crossner_entity_tsne_wt_er.png


python -m owner.visualize_entity_embeddings \
  --checkpoint checkpoints/owner/conll2003/soft_model/entity_typing.pt \
  --output outputs/crossner_entity_tsne_soft_model.png


python -m owner.visualize_entity_embeddings \
  --checkpoint checkpoints/ner/conll2003/100/entity_typing.pt \
  --output outputs/crossner_entity_tsne_old.png



cd WIN-thesis-template
xelatex WIN-thesis.tex && bibtex WIN-thesis && xelatex WIN-thesis.tex && xelatex WIN-thesis.tex



save_path = "checkpoints/owner/conll2003/test"


  python -m owner/visualize_entity_embeddings.py --checkpoint checkpoints/owner/new_model_2stage/entity_typing.pt --output outputs/soft_prompt.png


  conda activate owner   # if you use the project env
python -m owner.visualize_entity_embeddings \
  --checkpoint checkpoints/owner/new_model_2stage/entity_typing.pt \
  --output outputs/entity_tsne.png