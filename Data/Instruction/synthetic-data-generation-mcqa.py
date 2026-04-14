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
from prompts import MCQA_EVALUATION_PROMPT, MCQA_GENERATION_PROMPT, MCQA_REGENERATION_PROMPT  # noqa: E402

LLM_TEMPERATURE = 0.0
MAX_RETRIES = 2

_BASE_DIR = Path(__file__).parent
INPUT_PATH = _BASE_DIR / "../corpus/textbook/formatted-textbooks.xlsx"
OUTPUT_PATH = _BASE_DIR / "generated-mcqa-data.xlsx"

LLM_GENERATION_MODEL = "gpt-4o-mini"   # MCQA 생성 모델
LLM_EVALUATION_MODEL = "gpt-4o-mini"   # 평가 모델
LLM_REGENERATION_MODEL = "gpt-4o"      # 재생성 모델
llm_generation = ChatOpenAI(model=LLM_GENERATION_MODEL, temperature=LLM_TEMPERATURE)
llm_evaluation = ChatOpenAI(model=LLM_EVALUATION_MODEL, temperature=LLM_TEMPERATURE)
llm_regeneration = ChatOpenAI(model=LLM_REGENERATION_MODEL, temperature=LLM_TEMPERATURE)

# 유효한 MCQA 필수 키
_REQUIRED_KEYS = {"instruction", "option_a", "option_b", "answer", "explanation"}


class MCQAState(TypedDict):
    corpus_text: str        # 입력 corpus 텍스트
    qa_pairs: list          # [mcqa_dict, ...] — 생성된 전체 MCQA 목록
    qa_index: int           # 현재 처리 중인 MCQA 인덱스
    instruction: str        # 현재 처리 중인 질문
    option_a: str           # 선택지 A
    option_b: str           # 선택지 B
    option_c: str           # 선택지 C (없으면 빈 문자열)
    option_d: str           # 선택지 D (없으면 빈 문자열)
    option_e: str           # 선택지 E (없으면 빈 문자열)
    answer: str             # 정답 선택지 레이블 (a/b/c/d/e)
    explanation: str        # 정답 해설
    evaluation: dict        # {"answer": "o"/"x", "reason": str}
    retry_count: int        # 현재 MCQA의 재생성 횟수 (최대 MAX_RETRIES)
    accepted_pairs: list    # [mcqa_dict, ...] — 완료된 MCQA 누적


# 유틸리티
def parse_json_response(text: str):
    """LLM 응답 텍스트에서 JSON 객체/배열 추출. 코드블록 래핑 자동 처리."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)


def format_options_text(state: MCQAState) -> str:
    """선택지를 평가 프롬프트용 텍스트로 포맷."""
    lines = [
        f"a) {state['option_a']}",
        f"b) {state['option_b']}",
    ]
    if state["option_c"]:
        lines.append(f"c) {state['option_c']}")
    if state["option_d"]:
        lines.append(f"d) {state['option_d']}")
    if state["option_e"]:
        lines.append(f"e) {state['option_e']}")
    return "\n".join(lines)


def mcqa_dict_from_state(state: MCQAState) -> dict:
    """현재 state의 MCQA 필드를 dict로 변환."""
    return {
        "instruction": state["instruction"],
        "option_a": state["option_a"],
        "option_b": state["option_b"],
        "option_c": state["option_c"],
        "option_d": state["option_d"],
        "option_e": state["option_e"],
        "answer": state["answer"],
        "explanation": state["explanation"],
    }


def state_from_mcqa_dict(state: MCQAState, mcqa: dict) -> MCQAState:
    """MCQA dict를 state 필드에 반영."""
    return {
        **state,
        "instruction": mcqa.get("instruction", ""),
        "option_a": mcqa.get("option_a") or "",
        "option_b": mcqa.get("option_b") or "",
        "option_c": mcqa.get("option_c") or "",
        "option_d": mcqa.get("option_d") or "",
        "option_e": mcqa.get("option_e") or "",
        "answer": (mcqa.get("answer") or "").lower().strip(),
        "explanation": mcqa.get("explanation") or "",
    }


def is_valid_mcqa(mcqa: dict) -> bool:
    """필수 키 존재 여부 및 정답 레이블 유효성 검사."""
    if not _REQUIRED_KEYS.issubset(mcqa.keys()):
        return False
    answer = (mcqa.get("answer") or "").lower().strip()
    return answer in {"a", "b", "c", "d", "e"}


# LangGraph 노드 함수
def generate_all_mcqa(state: MCQAState) -> MCQAState:
    """① corpus_text에서 가능한 모든 MCQA를 한 번에 생성 (gpt-4o-mini)."""
    chain = MCQA_GENERATION_PROMPT | llm_generation
    response = chain.invoke({"corpus_text": state["corpus_text"]})

    try:
        qa_pairs = parse_json_response(response.content)
        if isinstance(qa_pairs, dict):
            qa_pairs = [qa_pairs]
        # 필수 키와 유효한 정답 레이블을 가진 항목만 유지
        qa_pairs = [p for p in qa_pairs if is_valid_mcqa(p)]
    except (json.JSONDecodeError, AttributeError, TypeError):
        qa_pairs = []

    return {
        **state,
        "qa_pairs": qa_pairs,
        "qa_index": 0,
        "instruction": "",
        "option_a": "", "option_b": "", "option_c": "", "option_d": "", "option_e": "",
        "answer": "",
        "explanation": "",
        "evaluation": {},
        "retry_count": 0,
        "accepted_pairs": [],
    }


def load_next_mcqa(state: MCQAState) -> MCQAState:
    """qa_pairs에서 다음 MCQA를 꺼내 현재 처리 대상으로 설정."""
    mcqa = state["qa_pairs"][state["qa_index"]]
    updated = state_from_mcqa_dict(state, mcqa)
    return {**updated, "evaluation": {}, "retry_count": 0}


def evaluate_mcqa(state: MCQAState) -> MCQAState:
    """② 5가지 기준 기반 MCQA 품질 평가 (gpt-4o-mini)."""
    chain = MCQA_EVALUATION_PROMPT | llm_evaluation
    response = chain.invoke({
        "instruction": state["instruction"],
        "options_text": format_options_text(state),
        "answer": state["answer"],
        "explanation": state["explanation"],
    })

    try:
        evaluation = parse_json_response(response.content)
        if "answer" not in evaluation:
            evaluation = {"answer": "o", "reason": ""}
    except (json.JSONDecodeError, AttributeError):
        evaluation = {"answer": "o", "reason": ""}

    return {**state, "evaluation": evaluation}


def regenerate_mcqa(state: MCQAState) -> MCQAState:
    """③ 평가 피드백을 반영하여 MCQA 전체 재생성 (gpt-4o)."""
    mcqa_json = json.dumps(mcqa_dict_from_state(state), ensure_ascii=False)
    evaluation_text = json.dumps(state["evaluation"], ensure_ascii=False)

    chain = MCQA_REGENERATION_PROMPT | llm_regeneration
    response = chain.invoke({
        "mcqa_json": mcqa_json,
        "evaluation": evaluation_text,
    })

    try:
        regenerated = parse_json_response(response.content)
        # 배열로 반환된 경우 첫 번째 항목만 사용
        if isinstance(regenerated, list):
            regenerated = regenerated[0]
        if not is_valid_mcqa(regenerated):
            regenerated = mcqa_dict_from_state(state)  # 실패 시 기존 유지
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
        regenerated = mcqa_dict_from_state(state)

    updated = state_from_mcqa_dict(state, regenerated)
    return {**updated, "retry_count": state["retry_count"] + 1}


def save_and_next(state: MCQAState) -> MCQAState:
    """현재 MCQA를 accepted_pairs에 저장하고 다음 인덱스로 이동."""
    updated_pairs = state["accepted_pairs"] + [{
        **mcqa_dict_from_state(state),
        "retry_count": state["retry_count"],
    }]
    return {
        **state,
        "accepted_pairs": updated_pairs,
        "qa_index": state["qa_index"] + 1,
    }


# 조건 엣지
def should_regenerate(state: MCQAState) -> Literal["regenerate", "accept"]:
    """평가 통과("o") 또는 최대 재시도 횟수 초과 → accept, 그 외 → regenerate."""
    eval_result = state["evaluation"].get("answer", "o")
    if eval_result == "o" or state["retry_count"] >= MAX_RETRIES:
        return "accept"
    return "regenerate"


def has_more_qa(state: MCQAState) -> Literal["load_next_mcqa", "__end__"]:
    """처리할 MCQA가 남아 있으면 load_next_mcqa, 모두 완료되면 END."""
    if state["qa_index"] < len(state["qa_pairs"]):
        return "load_next_mcqa"
    return "__end__"


# LangGraph 그래프 빌드
def build_graph():
    graph = StateGraph(MCQAState)

    graph.add_node("generate_all_mcqa", generate_all_mcqa)
    graph.add_node("load_next_mcqa", load_next_mcqa)
    graph.add_node("evaluate_mcqa", evaluate_mcqa)
    graph.add_node("regenerate_mcqa", regenerate_mcqa)
    graph.add_node("save_and_next", save_and_next)

    graph.add_edge(START, "generate_all_mcqa")
    graph.add_edge("generate_all_mcqa", "load_next_mcqa")
    graph.add_edge("load_next_mcqa", "evaluate_mcqa")
    graph.add_conditional_edges(
        "evaluate_mcqa",
        should_regenerate,
        {"regenerate": "regenerate_mcqa", "accept": "save_and_next"},
    )
    graph.add_edge("regenerate_mcqa", "evaluate_mcqa")
    graph.add_conditional_edges(
        "save_and_next",
        has_more_qa,
        {"load_next_mcqa": "load_next_mcqa", "__end__": END},
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
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MCQA 생성 중"):
        corpus_text = row["formatted"]

        # 초기 상태 설정
        initial_state: MCQAState = {
            "corpus_text": corpus_text,
            "qa_pairs": [],
            "qa_index": 0,
            "instruction": "",
            "option_a": "", "option_b": "", "option_c": "", "option_d": "", "option_e": "",
            "answer": "",
            "explanation": "",
            "evaluation": {},
            "retry_count": 0,
            "accepted_pairs": [],
        }

        # 파이프라인 실행
        final_state = pipeline.invoke(initial_state)

        # corpus 1개당 여러 MCQA → 각각 행으로 저장
        for pair in final_state["accepted_pairs"]:
            results.append({
                "source": row.get("file", ""),
                "page": row.get("page", ""),
                "corpus": corpus_text,
                "instruction": pair["instruction"],
                "option_a": pair["option_a"],
                "option_b": pair["option_b"],
                "option_c": pair["option_c"],
                "option_d": pair["option_d"],
                "option_e": pair["option_e"],
                "answer": pair["answer"],
                "explanation": pair["explanation"],
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
