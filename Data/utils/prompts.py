from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

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

# EQA QA 생성 프롬프트: corpus에서 가능한 모든 개방형 질문-답변 쌍 생성
QA_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a clinical ophthalmology educator creating a high-quality instruction dataset for training a medical language model.

Your task is to generate as many open-ended question-answer (EQA) pairs as possible from the provided ophthalmology textbook passage.

## Strategy — maximize coverage:
- Identify every distinct clinical concept in the passage: definitions, mechanisms, diagnostic criteria, treatments, anatomy, epidemiology, pathophysiology, differential diagnoses, prognosis, complications, etc.
- Generate one question per distinct concept — do not merge multiple concepts into a single question
- Vary question types: "What is...", "How does...", "What are the...", "Why does...", "Describe...", "What distinguishes..."

## Guidelines for each Question (instruction):
- Ask about clinically relevant knowledge
- Phrase the question clearly and precisely so it can be answered without reading the passage
- Avoid vague, trivial, or redundant questions

## Guidelines for each Answer:
- Provide a comprehensive, medically accurate answer based strictly on the passage
- Include all relevant clinical details, mechanisms, and context from the passage
- Write in clear, professional medical prose (no bullet points)
- Do not fabricate or add information not present in the passage

## Output Format (JSON array only, no other text):
[
  {{"instruction": "<question>", "answer": "<comprehensive answer>"}},
  {{"instruction": "<question>", "answer": "<comprehensive answer>"}},
  ...
]""",
    ),
    (
        "human",
        """Ophthalmology Textbook Passage:
{corpus_text}

Generate all possible EQA pairs as a JSON array:""",
    ),
])

# EQA 답변 평가 프롬프트: 5가지 규칙(일관성·관련성·맞춤응답·정확성·중립성) 준수 여부 판단
EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict quality evaluator for ophthalmology instruction data.

Evaluate the given answer to the given question according to the following 5 rules:

1. Consistency: Does the answer thoroughly consider the background or context of the question?
2. Relevance: Does the answer stay on topic and not mention content outside the scope of the question?
3. User-customized response: Does the answer address what was specifically asked (format, criteria, etc.)?
4. Accuracy: Does the answer contain only accurate, verified information without unverified or fabricated claims?
5. Neutrality: Does the answer use a neutral tone and avoid biased perspectives?

If ALL 5 rules are satisfied, set "answer" to "o" and "reason" to "".
If ANY rule is violated, set "answer" to "x" and "reason" to a concise explanation of which rules were violated and why.

## Output Format (JSON only, no other text):
{{"answer": "o or x", "reason": "<explanation if x, empty string if o>"}}""",
    ),
    (
        "human",
        """[Question]
{instruction}

[Answer]
{answer}

Evaluate:""",
    ),
])

# MCQA 생성 프롬프트: corpus에서 가능한 모든 객관식 문제 생성 (2지선다/4지선다/5지선다 혼합)
MCQA_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a clinical ophthalmology educator creating a high-quality multiple-choice question (MCQA) dataset for training a medical language model.

Your task is to generate as many MCQA problems as possible from the provided ophthalmology textbook passage.

## Strategy — maximize coverage:
- Identify every distinct clinical concept: definitions, mechanisms, diagnostic criteria, treatments, anatomy, epidemiology, pathophysiology, differentials, prognosis, complications
- Generate one question per distinct concept — do not merge multiple concepts into a single question
- Mix question types: True/False (2 choices), 4-choice, and 5-choice questions

## Guidelines for each Question:
- Focus on clinically relevant, testable knowledge
- Phrase clearly and precisely; the question must be answerable without reading the passage
- Avoid trivial, ambiguous, or redundant questions

## Guidelines for Options:
- For True/False: use exactly options "a" (True) and "b" (False)
- For 4-choice: use options "a", "b", "c", "d"
- For 5-choice: use options "a", "b", "c", "d", "e"
- All distractors must be plausible and clinically relevant (not obviously wrong)
- Only one option is correct; do not use "all of the above" or "none of the above"
- Use "option_e": null for questions with fewer than 5 choices

## Guidelines for the Explanation:
- Clearly justify why the correct answer is right, based strictly on the passage
- Where appropriate, briefly explain why the other options are incorrect
- Write in clear, professional medical prose (no bullet points)

## Output Format (JSON array only, no other text):
[
  {{
    "instruction": "<question>",
    "option_a": "<option A>",
    "option_b": "<option B>",
    "option_c": "<option C or null>",
    "option_d": "<option D or null>",
    "option_e": null,
    "answer": "<correct option letter: a/b/c/d/e>",
    "explanation": "<explanation>"
  }},
  ...
]""",
    ),
    (
        "human",
        """Ophthalmology Textbook Passage:
{corpus_text}

Generate all possible MCQA problems as a JSON array:""",
    ),
])

# MCQA 답변 평가 프롬프트: 정답 정확성·선택지 품질·해설 완성도 등 5가지 기준 평가
MCQA_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict quality evaluator for ophthalmology multiple-choice question (MCQA) data.

Evaluate the given MCQA problem according to the following 5 rules:

1. Accuracy: Is the marked answer truly correct based on medical knowledge? Are all other options incorrect?
2. Distractor quality: Are the wrong options plausible and clinically relevant (not obviously wrong or absurd)?
3. Explanation completeness: Does the explanation correctly justify the correct answer and, where appropriate, clarify why other options are wrong?
4. Relevance: Is the question relevant to ophthalmology and clearly testable?
5. Neutrality: Are the question and options unambiguously and neutrally worded, with no trick phrasing?

If ALL 5 rules are satisfied, set "answer" to "o" and "reason" to "".
If ANY rule is violated, set "answer" to "x" and "reason" to a concise explanation of which rules were violated and why.

## Output Format (JSON only, no other text):
{{"answer": "o or x", "reason": "<explanation if x, empty string if o>"}}""",
    ),
    (
        "human",
        """[Question]
{instruction}

[Options]
{options_text}

[Correct Answer]
{answer}

[Explanation]
{explanation}

Evaluate:""",
    ),
])

# MCQA 피드백 기반 재생성 프롬프트: 평가 피드백을 반영하여 문제 전체 개선 (논문 표 1 기반)
MCQA_REGENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are working on a task of creating multiple-choice questions about ophthalmology.
If the "answer" in [Evaluation] is "o", output [MCQA] as it is.
If it is "x", consider the content of "reason" in [Evaluation] and regenerate the entire MCQA according to [Rules].

[Rules]
1. Ensure the marked answer is factually correct and all other options are incorrect. (Accuracy)
2. Make all distractors plausible and clinically relevant — no obviously wrong options. (Distractor quality)
3. Provide a complete explanation that justifies the correct answer and, where appropriate, why other options are wrong. (Explanation completeness)
4. Keep the question clearly relevant to ophthalmology and testable. (Relevance)
5. Use unambiguous, neutral wording for the question and all options. (Neutrality)

Output the regenerated MCQA as a JSON object (same format as input), no other text.""",
    ),
    (
        "human",
        """[MCQA]
{mcqa_json}

[Evaluation]
{evaluation}

[Regenerated MCQA]""",
    ),
])

# EQA 피드백 기반 재생성 프롬프트: 평가 피드백을 반영하여 답변 개선 (논문 표 1 기반)
REGENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are working on a task of responding to questions about ophthalmology.
If the "answer" in [Evaluation] is "o", output [Answer] as it is.
If it is "x", consider the content of "reason" in [Evaluation] and re-answer [Question] according to [Rules].

[Rules]
1. Compose your answer by thoroughly considering the background or context of the question. (Consistency)
2. Do not mention content that falls outside the scope of the question and be careful to stay on topic. (Relevance)
3. If the user has requested a specific format or criteria, make sure to respond accordingly. (User-customized response)
4. Only mention accurate information in your answer and do not include unverified content. (Accuracy)
5. Use a neutral tone and expression to avoid biased perspectives. (Neutrality)""",
    ),
    (
        "human",
        """[Question]
{instruction}

[Answer]
{answer}

[Evaluation]
{evaluation}

[Re-answer]""",
    ),
])

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
