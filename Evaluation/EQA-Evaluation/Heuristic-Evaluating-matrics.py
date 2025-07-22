import pandas as pd
from transformers import AutoTokenizer
import nltk
nltk.download('wordnet')

excel_path = "./Evaluation/EQA-Evaluation/EQA-Ophthal-Results.xlsx"
df = pd.read_excel(excel_path)

ground_truth = df['answer']
model_output = df['collected_answer']

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import wordnet as wn
wn.ensure_loaded()
from nltk.translate import meteor_score
from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def Evaluation_rouge(tokenizer, sent1, sent2):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=tokenizer
    )
    rouge1 = scorer.score(sent1, sent2)['rouge1'].fmeasure
    rouge2 = scorer.score(sent1, sent2)['rouge2'].fmeasure
    rougeL = scorer.score(sent1, sent2)['rougeL'].fmeasure

    return rouge1, rouge2, rougeL

def Evaluation_bleu(tokenizer, sent1, sent2):
    bleu_score = sentence_bleu(
        [tokenizer.tokenize(sent1)],
        tokenizer.tokenize(sent2),
    )
    return bleu_score

def Evaluation_meteor(tokenizer, sent1, sent2):
    meteor_value = meteor_score.meteor_score(
        [tokenizer.tokenize(sent1)],
        tokenizer.tokenize(sent2),
    )
    return meteor_value

def Evaluation_semscore(sent1, sent2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding1 = model.encode(sent1, convert_to_tensor=True)
    embedding2 = model.encode(sent2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2).item()
    return cosine_scores

from tqdm import tqdm

metrics = {
    "rouge1": [],
    "rouge2": [],
    "rougeL": [],
    "bleu": [],
    "meteor": [],
    "semscore": []
}

for GT, Ans in tqdm(zip(ground_truth, model_output), total=len(ground_truth)):
    try:
        r1, r2, rL = Evaluation_rouge(tokenizer, GT, Ans)
    except Exception:
        r1, r2, rL = 0, 0, 0
    try:
        bleu = Evaluation_bleu(tokenizer, GT, Ans)
    except Exception:
        bleu = 0
    try:
        meteor = Evaluation_meteor(tokenizer, GT, Ans)
    except Exception:
        meteor = 0
    try:
        semscore = Evaluation_semscore(GT, Ans)
    except Exception:
        semscore = 0

    metrics["rouge1"].append(r1)
    metrics["rouge2"].append(r2)
    metrics["rougeL"].append(rL)
    metrics["bleu"].append(bleu)
    metrics["meteor"].append(meteor)
    metrics["semscore"].append(semscore)

for key in metrics:
    avg = sum(metrics[key]) / len(metrics[key]) if metrics[key] else 0
    print(f"{key.capitalize()}: {avg:.5f}")