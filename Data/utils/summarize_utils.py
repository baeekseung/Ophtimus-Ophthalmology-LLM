"""
요약 유틸리티 모듈
- stuff_summarize: 전체 텍스트를 한 번에 요약 (context length가 짧을 때)
- map_reduce_summarize: 페이지별로 요약 후 합산 (context length가 길 때)
"""

from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from prompts import MAP_PROMPT, REDUCE_PROMPT, STUFF_PROMPT, TEXTBOOK_PAGE_PROMPT  # noqa: F401


def stuff_summarize(full_text: str, llm) -> str:
    """
    Stuff 방식: 논문 전체 텍스트를 한 번에 LLM에 전달하여 pre-training용 정제 텍스트 생성.
    context length가 70K 이하인 경우에 사용.
    """
    docs = [Document(page_content=full_text)]
    chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=STUFF_PROMPT,
        verbose=False,
    )
    result = chain.invoke({"input_documents": docs})
    return result["output_text"]


def textbook_page_format(page_text: str, llm) -> str:
    """
    Stuff 방식: 교과서 단일 페이지 텍스트를 LLM에 전달하여 정제된 산문으로 변환.
    페이지 단위로 호출하며, 빈 페이지는 호출 전 필터링 권장.
    """
    docs = [Document(page_content=page_text)]
    chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=TEXTBOOK_PAGE_PROMPT,
        verbose=False,
    )
    result = chain.invoke({"input_documents": docs})
    return result["output_text"]


def map_reduce_summarize(pages: list[str], llm) -> str:
    """
    Map-Reduce 방식: 페이지 단위로 개별 추출(map)한 후 통합 정제(reduce).
    context length가 70K 초과인 경우에 사용.
    """
    docs = [Document(page_content=page) for page in pages if page.strip()]
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=REDUCE_PROMPT,
        verbose=False,
    )
    result = chain.invoke({"input_documents": docs})
    return result["output_text"]
