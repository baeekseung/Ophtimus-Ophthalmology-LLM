import json
import re
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from tqdm import tqdm
from typing_extensions import TypedDict

sys.path.append(str(Path(__file__).parent.parent.parent / "Data" / "utils"))
from prompts import EVALUATION_PROMPT, QA_GENERATION_PROMPT, REGENERATION_PROMPT  # noqa: E402

LLM_TEMPERATURE = 0.0
MAX_RETRIES = 2

_BASE_DIR = Path(__file__).parent
INPUT_PATH = _BASE_DIR / "../corpus/textbook/formatted-textbooks.xlsx"
OUTPUT_PATH = _BASE_DIR / "generated-eqa-data.xlsx"

LLM_GENERATION_MODEL = "gpt-4o-mini"   # QA 생성 모델
LLM_EVALUATION_MODEL = "gpt-4o-mini"   # 평가 모델
LLM_REGENERATION_MODEL = "gpt-4o"      # 재생성 모델
llm_generation = ChatOpenAI(model=LLM_GENERATION_MODEL, temperature=LLM_TEMPERATURE)
llm_evaluation = ChatOpenAI(model=LLM_EVALUATION_MODEL, temperature=LLM_TEMPERATURE)
llm_regeneration = ChatOpenAI(model=LLM_REGENERATION_MODEL, temperature=LLM_TEMPERATURE)


class SyntheticDataState(TypedDict):
    corpus_text: str        # 입력 corpus 텍스트
    qa_pairs: list          # [{instruction, answer}, ...] — 생성된 전체 QA 목록
    qa_index: int           # 현재 처리 중인 QA 인덱스
    instruction: str        # 현재 처리 중인 질문
    answer: str             # 현재 처리 중인 답변
    evaluation: dict        # {"answer": "o"/"x", "reason": str}
    retry_count: int        # 현재 QA의 재생성 횟수 (최대 MAX_RETRIES)
    accepted_pairs: list    # [{instruction, answer, retry_count}, ...] — 완료된 QA 누적


def parse_json_response(text: str):
    """LLM 응답 텍스트에서 JSON 객체/배열 추출. 코드블록 래핑 자동 처리."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)

# LangGraph 노드 함수
def generate_all_qa(state: SyntheticDataState) -> SyntheticDataState:
    """① corpus_text에서 가능한 모든 QA 쌍을 한 번에 생성 (gpt-4o-mini)."""
    chain = QA_GENERATION_PROMPT | llm_generation
    response = chain.invoke({"corpus_text": state["corpus_text"]})

    try:
        qa_pairs = parse_json_response(response.content)
        # 단일 dict로 반환된 경우 리스트로 감싸기
        if isinstance(qa_pairs, dict):
            qa_pairs = [qa_pairs]
        # instruction/answer 키를 가진 항목만 유지
        qa_pairs = [p for p in qa_pairs if p.get("instruction") and p.get("answer")]
    except (json.JSONDecodeError, AttributeError, TypeError):
        qa_pairs = []

    return {
        **state,
        "qa_pairs": qa_pairs,
        "qa_index": 0,
        "instruction": "",
        "answer": "",
        "evaluation": {},
        "retry_count": 0,
        "accepted_pairs": [],
    }


def load_next_qa(state: SyntheticDataState) -> SyntheticDataState:
    """qa_pairs에서 다음 QA를 꺼내 현재 처리 대상으로 설정."""
    pair = state["qa_pairs"][state["qa_index"]]
    return {
        **state,
        "instruction": pair["instruction"],
        "answer": pair["answer"],
        "evaluation": {},
        "retry_count": 0,
    }


def evaluate_answer(state: SyntheticDataState) -> SyntheticDataState:
    """② 5가지 규칙 기반 답변 품질 평가 (gpt-4o-mini)."""
    chain = EVALUATION_PROMPT | llm_evaluation
    response = chain.invoke({
        "instruction": state["instruction"],
        "answer": state["answer"],
    })

    try:
        evaluation = parse_json_response(response.content)
        if "answer" not in evaluation:
            evaluation = {"answer": "o", "reason": ""}
    except (json.JSONDecodeError, AttributeError):
        evaluation = {"answer": "o", "reason": ""}

    return {**state, "evaluation": evaluation}


def regenerate_answer(state: SyntheticDataState) -> SyntheticDataState:
    """③ 평가 피드백을 반영하여 답변 재생성 (gpt-4o)."""
    evaluation_text = json.dumps(state["evaluation"], ensure_ascii=False)
    chain = REGENERATION_PROMPT | llm_regeneration
    response = chain.invoke({
        "instruction": state["instruction"],
        "answer": state["answer"],
        "evaluation": evaluation_text,
    })

    return {
        **state,
        "answer": response.content.strip(),
        "retry_count": state["retry_count"] + 1,
    }


def save_and_next(state: SyntheticDataState) -> SyntheticDataState:
    """현재 QA를 accepted_pairs에 저장하고 다음 QA 인덱스로 이동."""
    updated_pairs = state["accepted_pairs"] + [{
        "instruction": state["instruction"],
        "answer": state["answer"],
        "retry_count": state["retry_count"],
    }]
    return {
        **state,
        "accepted_pairs": updated_pairs,
        "qa_index": state["qa_index"] + 1,
    }


# 조건 엣지
def should_regenerate(state: SyntheticDataState) -> Literal["regenerate", "accept"]:
    """평가 통과("o") 또는 최대 재시도 횟수 초과 → accept, 그 외 → regenerate."""
    eval_result = state["evaluation"].get("answer", "o")
    if eval_result == "o" or state["retry_count"] >= MAX_RETRIES:
        return "accept"
    return "regenerate"


def has_more_qa(state: SyntheticDataState) -> Literal["load_next_qa", "__end__"]:
    """처리할 QA가 남아 있으면 load_next_qa, 모두 완료되면 END."""
    if state["qa_index"] < len(state["qa_pairs"]):
        return "load_next_qa"
    return "__end__"

# LangGraph 그래프 빌드
def build_graph():
    graph = StateGraph(SyntheticDataState)

    graph.add_node("generate_all_qa", generate_all_qa)
    graph.add_node("load_next_qa", load_next_qa)
    graph.add_node("evaluate_answer", evaluate_answer)
    graph.add_node("regenerate_answer", regenerate_answer)
    graph.add_node("save_and_next", save_and_next)

    graph.add_edge(START, "generate_all_qa")
    graph.add_edge("generate_all_qa", "load_next_qa")
    graph.add_edge("load_next_qa", "evaluate_answer")
    graph.add_conditional_edges(
        "evaluate_answer",
        should_regenerate,
        {"regenerate": "regenerate_answer", "accept": "save_and_next"},
    )
    graph.add_edge("regenerate_answer", "evaluate_answer")
    graph.add_conditional_edges(
        "save_and_next",
        has_more_qa,
        {"load_next_qa": "load_next_qa", "__end__": END},
    )

    return graph.compile()



# 메인 실행
def main():
    # 입력 데이터 로드
    print(f"[INFO] 입력 파일 로드: {INPUT_PATH.resolve()}")
    df = pd.read_excel(INPUT_PATH)

    # 유효한 formatted 텍스트만 필터링 (빈 값, '콘텐츠 없음' 제외)
    df = df[
        df["formatted"].notna()
        & (df["formatted"].str.strip() != "")
        & (df["formatted"] != "콘텐츠 없음")
    ].reset_index(drop=True)
    print(f"[INFO] 처리 대상 행 수: {len(df)}")

    # 그래프 빌드
    pipeline = build_graph()

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="EQA 생성 중"):
        corpus_text = row["formatted"]

        # 초기 상태 설정
        initial_state: SyntheticDataState = {
            "corpus_text": corpus_text,
            "qa_pairs": [],
            "qa_index": 0,
            "instruction": "",
            "answer": "",
            "evaluation": {},
            "retry_count": 0,
            "accepted_pairs": [],
        }

        # 파이프라인 실행
        final_state = pipeline.invoke(initial_state)

        # corpus 1개당 여러 QA → 각각 행으로 저장
        for pair in final_state["accepted_pairs"]:
            results.append({
                "source": row.get("file", ""),
                "page": row.get("page", ""),
                "corpus": corpus_text,
                "instruction": pair["instruction"],
                "answer": pair["answer"],
                "retry_count": pair["retry_count"],
            })

    # 결과 저장
    result_df = pd.DataFrame(results)
    result_df.to_excel(OUTPUT_PATH, index=False)

    regenerated_count = (result_df["retry_count"] > 0).sum()
    print(f"[INFO] 저장 완료: {OUTPUT_PATH.resolve()}")
    print(f"[INFO] 총 {len(result_df)}개 | 재생성 발생: {regenerated_count}개")


if __name__ == "__main__":
    main()
