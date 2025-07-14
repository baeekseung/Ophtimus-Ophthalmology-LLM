from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_teddynote import logging
logging.langsmith("Map-Reduce-process")
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that summarizes documents."),
        ("user", "{instruction}"),
    ]
)

def create_map_instruction(docs):
    return f"""The following is a partial content of the ophthalmology paper.
    {docs}
    Write a summary of the given content, focusing on information related to eye diseases and treatments. Do not include information about the authors and references. If the information is not relevant to eye diseases and treatments, don't include it in your summary.
    Answer:"""

def create_reduce_instruction(docs):
    return f"""The following is a Series of Ophthalmology Paper Summaries.
    {docs}
    Please generate a detailed summary of the following text in multiple paragraphs without using headings or subheadings, focusing on the content related to eye diseases and treatments. Please summarize in as much detail as possible.
    Answer:"""

chain = (
    {"instruction": RunnableLambda(create_map_instruction)}
    | prompt
    | llm
    | StrOutputParser()
)







