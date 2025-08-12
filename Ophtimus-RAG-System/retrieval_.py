import logging
import pandas as pd

import pickle

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith("RAG-Data-ReFiner", set_enable=True)

embedding_model_name = "ibm-granite/granite-embedding-125m-english"

hf_embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

loaded_db = FAISS.load_local(
    folder_path="./Ophthalmology_FAISS_Index_final", 
    index_name="Ophthalmology_FAISS_Index",
    embeddings=hf_embeddings,
    allow_dangerous_deserialization=True,
    )

with open("./Ophtimus-RAG-System/bm25_retriever_final.pkl", "rb") as f:
    loaded_bm25_retriever = pickle.load(f)

loaded_bm25_retriever.k = 2


faiss_retriever = loaded_db.as_retriever(search_kwargs={"k": 2})

ensemble_retriever = EnsembleRetriever(
    retrievers=[loaded_bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5],
)

prompt = PromptTemplate(
    template="""# Your role
You are a medical assistant specialized in ophthalmology. Your role is to revise a model-generated answer to a question about ophthalmology, based solely on the provided reference documents related to the question.

# Instruction
Carefully review the entire model-generated answer — including any reasoning or thought process enclosed in <think> tags — and revise it using only the provided reference documents. Correct or remove any hallucinated or inaccurate information, both in the <think> section and the final written answer. If the original answer is missing important details or includes vague statements, supplement it with accurate information grounded in the references. 
Whenever possible, retain content that is not entirely incorrect, even if it requires slight rewording or clarification, rather than removing it outright.
If the content of the <model-generated answer> (including <think>) is completely accurate and requires no improvements, clearly state:
"No revisions needed."

<question>
Question:
{question}
</question>

<reference documents>
Reference Documents:
{context}
</reference documents>

<model-generated answer>
Model-Generated Answer:
{answer}
</model-generated answer>

# Constraint
1. Maintain a medically professional and objective tone throughout the answer.
2. Revise both the <think> section and the final answer when needed.
3. Whenever possible, preserve content that is partially correct by refining it instead of deleting it, unless it is factually incorrect.
4. Do not cite or list the reference documents explicitly in the answer.
5. Do not include phrases such as “according to the document” or “based on the reference.”
6. If multiple facts are supported, synthesize them into a cohesive and concise statement.
7. Do not rephrase the question or model-generated answer unless necessary for clarity.
8. If modifications have been made, please output the entire contents of the modified Model-Generated Answer (including <think> if present).

<refined model-generated answer>
Refined Model-Generated Answer:
""",
    input_variables=["context", "question", "answer"]
)

import pandas as pd

deepseek_df = pd.read_excel('./Ophtimus-RAG-System/Dx_DeepSeek_inference_results.xlsx', sheet_name='Sheet1')
deepseek_question = deepseek_df['instruction'].tolist()
deepseek_answer = deepseek_df['result'].tolist()

llm = ChatOpenAI(
    model_name="gpt-4o", 
    temperature=0
)

def make_chain_with_question(question):
    def extract_context(q):
        # ensemble_retriever가 question을 입력받아 Documents 객체 리스트 반환한다고 가정
        docs = ensemble_retriever.invoke(q)
        # 각 Document의 page_content만 추출하여 하나의 문자열로 합침
        return "\n".join([doc.page_content for doc in docs])
    
    return (
        {
            "context": lambda _: extract_context(question),  # context에 page_content 문자열 전달
            "question": lambda _: question,
            "answer": lambda inputs: inputs["answer"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

def extract_assistant_response(text: str) -> str:
    marker = "<｜Assistant｜>"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    else:
        return ""

results = []
checkpoint_interval = 100  # 300개마다 저장

for idx in range(len(deepseek_question)):
    print(f"{idx + 1}번째 질문 처리 중...")
    question = deepseek_question[idx]
    answer_raw = deepseek_answer[idx]
    input_answer = extract_assistant_response(answer_raw)
    chain = make_chain_with_question(question)
    try:
        chain_response = chain.invoke({"answer": input_answer})
    except Exception as e:
        chain_response = f"Error: {e}"
    results.append({
        "question": question,
        "deepseek_answer": answer_raw,
        "chain_response": chain_response
    })
    # 체크포인트 저장
    if (idx + 1) % checkpoint_interval == 0:
        output_filename = f'./Ophtimus-RAG-System/Dx_DeepSeek_chain_results_{idx + 1}.xlsx'
        temp_df = pd.DataFrame(results)
        temp_df.to_excel(output_filename, index=False)
        print(f"{idx + 1}개 저장 완료.")

# 마지막까지 저장
final_df = pd.DataFrame(results)
output_filename = f'./Ophtimus-RAG-System/Dx_DeepSeek_chain_results.xlsx'
final_df.to_excel(output_filename, index=False)
print("모든 결과 저장 완료.")