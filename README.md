# Phi3KnowChem
Submission to the L+M-24 shared task at ACL2024 Language + Molecule Workshop.
We trained Phi-3-mini-4k for the molecule captioning task.
Our work shows that the continued pretraining phase without direct exposure to SMILES representations significantly enhanced the model's performance, a 300\% increase for the BLEU scores. 

## Model Weight
The model weight can be found at 🔽[Hugging Face](https://huggingface.co/bluesky333/Phi3KnowChem).

## Evaluation Dataset Download
The dataset used for evaluation can be found at [LPM Dataset](https://github.com/language-plus-molecules/LPM-24-Dataset).

## Running Evaluation
You can generate captions/descriptions with the code in this repository.

```
git clone https://github.com/bluesky333/Phi3KnowChem
cd Phi3KnowChem
conda create -n knowchem python=3.10 -y
conda activate knowchem
pip install -r requirements.txt
```

🏃 Inference
```
python inference-caption.py -c bluesky333/Phi3KnowChem -d Phi3KnowChem -o Phi3KnowChem --max-seq-len 2048 --batch-size 1
```
