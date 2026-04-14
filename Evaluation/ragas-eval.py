from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
)

# 1. 모델 설정
LLM_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

METRICS_TO_USE = [
    "faithfulness",
    "answer_relevancy",
    "context_relevancy",
    "context_recall",]


def build_metric_list(metric_names: List[str]):
    name_to_metric = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_relevancy": context_relevancy,
        "context_recall": context_recall,
    }
    metrics = []
    for name in metric_names:
        if name not in name_to_metric:
            raise ValueError(f"알 수 없는 metric 이름입니다: {name}")
        metrics.append(name_to_metric[name])
    return metrics


# 2. 데이터: 질문/답변/컨텍스트/정답 수동 작성
def build_sample_data():
    import pandas as pd
    df = pd.read_csv("my_dataset.csv")
    return {
        "question": df["user_query"].tolist(),
        "answer": df["rag_answer"].tolist(),
        "contexts": df["retrieved_chunks"].tolist(),  # List[List[str]] 형태여야 함
        "ground_truth": df["label_answer"].tolist(),
    }



# 3. 평가 실행 함수
def run_local_ragas_eval():
    # 1) 데이터 준비
    data_dict = build_sample_data()
    dataset = Dataset.from_dict(data_dict)

    # 2) 메트릭 준비
    metrics = build_metric_list(METRICS_TO_USE)

    # (참고)
    # - 여기서는 evaluate()에 LLM/임베딩을 직접 넘기지 않고,
    #   ragas 내부 설정(예: OpenAI, Anthropic, Ollama 등 provider 설정)을 사용합니다.
    # - LLM_MODEL_NAME / EMBEDDING_MODEL_NAME는 실제론
    #   provider 설정에서 사용되도록 맞춰주시면 됩니다.
    print(f"Using LLM model = {LLM_MODEL_NAME}")
    print(f"Using embedding model = {EMBEDDING_MODEL_NAME}")
    print(f"Using metrics = {METRICS_TO_USE}")

    # 3) 평가 실행
    result = evaluate(dataset, metrics=metrics)

    # 4) 결과 출력
    print("\n=== 전체 평균 점수 ===")
    print(result)

    print("\n=== 샘플별 점수 (pandas) ===")
    df = result.to_pandas()
    print(df)


if __name__ == "__main__":
    run_local_ragas_eval()