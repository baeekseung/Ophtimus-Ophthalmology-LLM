"""
요약 유틸리티 모듈
- stuff_summarize: 전체 텍스트를 한 번에 요약 (context length가 짧을 때)
- map_reduce_summarize: 페이지별로 요약 후 합산 (context length가 길 때)
"""

from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# Stuff 방식: 논문 전체를 한 번에 처리하여 pre-training용 정제 텍스트 생성
STUFF_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a specialized medical knowledge curator responsible for extracting high-quality pre-training data from ophthalmology literature. Your output will be used to train a medical language model, so accuracy, clarity, and strict filtering are essential.

## Your Objective
Extract and rewrite ALL verified, generalizable ophthalmology knowledge from the provided paper — maximize the volume of qualifying content retained. Every sentence that meets the quality criteria below must be included; do not omit or compress qualifying content unnecessarily. At the same time, never add, infer, or generalize any information that is not explicitly stated in the source text. Discard only what does not meet the quality criteria below.

---

## Step-by-Step Process

**Step 1 — Identify qualifying content**
Scan the text for the following types of verified medical knowledge:
- Established anatomy and physiology of the eye and visual system
- Clinically validated diagnostic criteria, signs, and examination findings
- Evidence-based treatment protocols, medications, and therapeutic guidelines
- Well-established epidemiology, prevalence, and risk factors
- Recognized pathophysiology of ophthalmic conditions
- Standard surgical or procedural descriptions with accepted clinical outcomes

**Step 2 — Filter out all non-qualifying content (exclude without exception)**
- Author names, institutional affiliations, acknowledgments, funding disclosures
- Document identifiers: DOI, PMID, volume, issue, page numbers, copyright notices
- Reference lists, in-text citations (e.g., "[1]", "(Smith et al., 2020)"), and bibliographies
- Descriptions of or references to figures, tables, or graphs (e.g., "As shown in Figure 2...", "Table 1 demonstrates...")
- Experimental-only or study-specific findings not yet adopted into clinical practice
- Speculative or hedged language: "may suggest," "could potentially," "we hypothesize," "appears to," "it is possible that"
- Raw statistical reporting (p-values, confidence intervals, sample sizes) presented without generalizable clinical meaning
- Any content unrelated to ophthalmology

**Step 3 — Rewrite as clean natural language prose**
Transform all qualifying content into coherent, flowing paragraphs:
- Write in English
- Use no headings, titles, section labels, or bullet points of any kind
- Every sentence must end with a period
- Do not copy sentences verbatim; rephrase into clear, declarative medical prose
- Preserve the full detail of every qualifying passage — do not summarize or condense what can be kept in full
- Maintain strict factual accuracy; never add, infer, or extrapolate any information not explicitly stated in the source text
- Group related concepts into logical paragraphs that flow naturally from one to the next

**Step 4 — Self-verify before producing final output**
Check your draft against each criterion below. Fix any issue before finalizing:
□ Does the output contain author names, affiliations, or document identifiers? → Remove them
□ Does the output contain headings, bullet points, or lists? → Rewrite as prose
□ Does any sentence fail to end with a period? → Fix it
□ Does the output include speculative or unverified claims? → Remove them
□ Does the output reference figures, tables, or graphs? → Remove those sentences
□ Is there no qualifying content at all? → Output exactly: 콘텐츠 없음

---

## Output Rules
- Output ONLY the curated prose paragraphs, with no preamble or commentary
- Maximize the length and detail of the output — include every qualifying sentence from the source
- Never introduce facts, explanations, or context that are not present in the source text
- If no qualifying content exists, output exactly: 콘텐츠 없음

---

Paper Text:
{text}

Curated Medical Knowledge:""",
)

# Map 단계: 페이지(섹션) 단위로 pre-training에 적합한 의학 지식 추출
MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a medical content extractor building a pre-training dataset from ophthalmology papers. Your task is to identify and extract only verified, generalizable ophthalmology knowledge from the provided text section.

## EXTRACT — include only:
- Established eye anatomy, physiology, and pathophysiology
- Clinically validated diagnostic procedures, criteria, or examination findings
- Evidence-based treatment approaches, medications, or surgical techniques
- Widely accepted epidemiological facts and risk factors for ophthalmic conditions

Extract as much qualifying content as possible — do not omit or compress any passage that meets the criteria above. Never add, infer, or extrapolate any information that is not explicitly stated in the source text.

## DISCARD — exclude entirely:
- Author names, institutional affiliations, acknowledgments, or funding information
- Document identifiers, journal metadata, copyright notices
- Any reference to or description of figures, tables, or graphs
- Speculative or hedged statements ("may," "might," "could suggest," "we hypothesize")
- Raw statistical data (p-values, confidence intervals, sample sizes) without clinical context
- In-text citations and reference lists (e.g., "[3]", "(Jones et al., 2019)")
- Content unrelated to ophthalmology

## Format:
- Write in English as flowing, natural language prose
- No headings, section labels, or bullet points
- Every sentence must end with a period
- If this section contains no qualifying content, output exactly: 콘텐츠 없음

---

Section Text:
{text}

Extracted Medical Knowledge:""",
)

# Reduce 단계: 페이지별 추출 결과를 하나의 정제된 텍스트로 통합
REDUCE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a senior medical knowledge editor compiling verified ophthalmology content for LLM pre-training. You have received extracted knowledge passages from multiple sections of the same paper. Synthesize them into a single, coherent, deduplicated body of medical knowledge.

## Synthesis Instructions

**Include:**
- All established ophthalmological facts, mechanisms, and clinical knowledge present in the passages
- Content that is clinically validated and generalizable beyond this specific study
- Information from all non-empty sections, integrated into a logical narrative flow
- Preserve the full detail of every qualifying passage — maximize the total volume of retained content without omitting or condensing qualifying information
- Never add, infer, or introduce any fact or explanation that is not explicitly present in the provided passages

**Exclude:**
- Any remaining author names, affiliations, or document identifiers (even partial)
- Duplicate or redundant information — retain only the most complete version
- Sections or sentences marked "콘텐츠 없음" — skip them entirely
- Speculative, hedged, or study-specific claims
- Any bullet points or lists encountered — rewrite as prose

**Format:**
- Output a single, continuous body of text composed of coherent paragraphs
- Write in English
- No headings, titles, section labels, or bullet points of any kind
- Every sentence must end with a period
- Paragraphs should transition logically and read as a unified medical document
- If ALL input sections are "콘텐츠 없음" or contain no qualifying content, output exactly: 콘텐츠 없음

---

Extracted Sections:
{text}

Final Curated Medical Knowledge:""",
)



def stuff_summarize(full_text: str, llm: ChatOpenAI) -> str:
    """
    Stuff 방식: 논문 전체 텍스트를 한 번에 LLM에 전달하여 pre-training용 정제 텍스트 생성.
    context length가 70K 이하인 경우에 사용.

    Args:
        full_text: 논문 전체 텍스트 (페이지를 합친 문자열)
        llm: ChatOpenAI 인스턴스

    Returns:
        정제된 자연어 산문 텍스트
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


# Textbook 페이지 정형화: 교과서 단일 페이지를 정제된 산문으로 변환
TEXTBOOK_PAGE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a specialized medical knowledge curator extracting high-quality pre-training data from an ophthalmology textbook. Your output will be used to train a medical language model, so accuracy, completeness, and strict filtering are essential.

## Your Objective
Extract and rewrite ALL verified, educational ophthalmology knowledge from this textbook page — maximize the volume of qualifying content retained. Every sentence that meets the quality criteria below must be included; do not omit or compress qualifying content unnecessarily. Never add, infer, or generalize any information not explicitly stated in the source text.

---

## Step-by-Step Process

**Step 1 — Identify qualifying content**
Extract the following types of established medical knowledge:
- Definitions and classifications of ocular diseases, structures, or conditions
- Anatomy and physiology of the eye, orbit, and visual pathway
- Clinically established pathophysiology and disease mechanisms
- Diagnostic criteria, clinical signs, symptoms, and examination findings
- Evidence-based treatment guidelines, pharmacotherapy, and surgical procedures
- Established epidemiology, prevalence, and risk factors
- Standard clinical decision-making frameworks and differential diagnoses

**Step 2 — Filter out all non-qualifying content (exclude without exception)**
- Running headers and footers (e.g., chapter titles, section names repeated at the top/bottom of a page)
- Page numbers, chapter numbers, and any standalone numeric identifiers
- Figure captions and descriptions (e.g., "Figure 8-1.", "See Figure 3 for...", "As illustrated in...")
- Table titles, column headers, and tabular data without narrative context
- Reference lists, footnotes, and in-text citations (e.g., "[1]", "(Smith, 2020)")
- Any content clearly unrelated to ophthalmology

**Step 3 — Rewrite as clean natural language prose**
Transform all qualifying content into coherent, flowing paragraphs:
- Write in English
- Use no headings, titles, section labels, or bullet points of any kind
- Every sentence must end with a period
- Do not copy sentences verbatim; rephrase into clear, declarative medical prose
- Preserve the full detail of every qualifying passage — do not summarize or condense what can be kept in full
- Maintain strict factual accuracy; never add, infer, or extrapolate information not present in the source text
- Group related concepts into logical paragraphs that flow naturally from one to the next

**Step 4 — Self-verify before producing final output**
Check your draft against each criterion:
□ Does the output contain page numbers, chapter headers, or running footers? → Remove them
□ Does the output reference figures, tables, or graphs? → Remove those sentences
□ Does the output contain headings, bullet points, or lists? → Rewrite as prose
□ Does any sentence fail to end with a period? → Fix it
□ Is there no qualifying content at all? → Output exactly: 콘텐츠 없음

---

## Output Rules
- Output ONLY the curated prose paragraphs, with no preamble or commentary
- Maximize the length and detail of the output — include every qualifying sentence from the source
- Never introduce facts, explanations, or context that are not present in the source text
- If no qualifying content exists, output exactly: 콘텐츠 없음

---

Textbook Page Text:
{text}

Curated Medical Knowledge:""",
)


def textbook_page_format(page_text: str, llm: ChatOpenAI) -> str:
    """
    Stuff 방식: 교과서 단일 페이지 텍스트를 LLM에 전달하여 정제된 산문으로 변환.
    페이지 단위로 호출하며, 빈 페이지는 호출 전 필터링 권장.

    Args:
        page_text: 교과서 단일 페이지 텍스트
        llm: ChatOpenAI 인스턴스

    Returns:
        정제된 자연어 산문 텍스트 (내용 없으면 '콘텐츠 없음')
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


def map_reduce_summarize(pages: list[str], llm: ChatOpenAI) -> str:
    """
    Map-Reduce 방식: 페이지 단위로 개별 추출(map)한 후 통합 정제(reduce).
    context length가 70K 초과인 경우에 사용.

    Args:
        pages: 논문 페이지별 텍스트 리스트
        llm: ChatOpenAI 인스턴스

    Returns:
        정제된 통합 자연어 산문 텍스트
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
