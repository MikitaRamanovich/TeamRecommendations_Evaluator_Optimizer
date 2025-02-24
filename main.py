import os
import zipfile
import docx
import pandas as pd
import json
import pdfplumber
from typing import Literal
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task

from dotenv import load_dotenv
load_dotenv()
open_ai_api_key = os.getenv("OPENAI_API_KEY")
lang_smith_tracing = os.getenv("LANGSMITH_TRACING")
lang_smith_endpoint = os.getenv("LANGSMITH_ENDPOINT")
lang_smith_api_key = os.getenv("LANGSMITH_API_KEY")
lang_smith_project_name = os.getenv("LANGSMITH_PROJECT")

# --- Environment Setup ---
os.environ["OPENAI_API_KEY"] = open_ai_api_key
os.environ["LANGSMITH_TRACING"] = lang_smith_tracing
os.environ["LANGSMITH_ENDPOINT"] = lang_smith_endpoint
os.environ["LANGSMITH_API_KEY"] = lang_smith_api_key
os.environ["LANGSMITH_PROJECT"] = lang_smith_project_name

# --- Evaluator Schema and LLM Setup ---
class AnalysisFeedback(BaseModel):
    grade: Literal["acceptable", "not acceptable"] = Field(
        description="Grade the generated JSON response."
    )
    feedback: str = Field(
        description="If the response is not acceptable, provide feedback on how to improve it."
    )

# Create a global LLM instance and an evaluator augmented with structured output.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluator = llm.with_structured_output(AnalysisFeedback)

# --- Utility Functions for File Extraction ---
def extract_text_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        print(f"Error extracting DOCX from {file_path}: {e}")
        return ""

def extract_text_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting PDF from {file_path}: {e}")
    return text

def extract_text_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df.to_csv(index=False)
    except Exception as e:
        print(f"Error extracting Excel from {file_path}: {e}")
        return ""

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return extract_text_docx(file_path)
    elif ext == ".pdf":
        return extract_text_pdf(file_path)
    elif ext in [".xlsx", ".xls"]:
        return extract_text_excel(file_path)
    else:
        return ""

def process_folder(folder_path):
    """
    Traverses the folder (and subdirectories). If a zip archive is found, it extracts it and processes its content.
    """
    all_texts = []
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            temp_folder = os.path.join(root, "temp_extracted")
                            os.makedirs(temp_folder, exist_ok=True)
                            zip_ref.extractall(temp_folder)
                            extracted_text = process_folder(temp_folder)
                            all_texts.append(extracted_text)
                    except Exception as e:
                        print(f"Error processing zip file {file_path}: {e}")
                else:
                    text = extract_text_from_file(file_path)
                    if text:
                        print(f"Extracted text from {file_path} (length {len(text)} characters)")
                        all_texts.append(text)
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
    return "\n".join(all_texts)

def split_text(text, chunk_size=1000, chunk_overlap=500):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)
    except Exception as e:
        print(f"Error splitting text: {e}")
        return []

def build_vector_store(chunks):
    try:
        embeddings = OpenAIEmbeddings()  # Or choose another embedding provider
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error building vector store: {e}")
        return None

def retrieve_relevant_chunks(vectorstore, query, k=5):
    try:
        return vectorstore.similarity_search(query, k=k)
    except Exception as e:
        print(f"Error retrieving relevant chunks: {e}")
        return []

def generate_json_response(retrieved_chunks, user_question, feedback=None):
    """
    Generates a JSON response using the retrieved context and optional feedback.
    """
    context = "\n".join([doc.page_content for doc in retrieved_chunks])
    feedback_text = f"Feedback on previous answer: {feedback}\n\n" if feedback else ""
    prompt_template = (
        "You are an expert project analyst. Analyze the following project description "
        "and provide a recommendation for the necessary team composition to develop this project. "
        "Based on the following project description:\n"
        "{context}\n\n"
        "{feedback_text}"
        "Answer the following question:\n"
        "{question}\n\n"
        "Generate a response in valid JSON format with the following structure exactly, and return only the JSON object without any additional text or markdown formatting:\n\n"
        "{{\n"
        '  "project_description": "<a concise summary of the project>",\n'
        '  "team_recommendation": [\n'
        "    {{\n"
        '      "role": "<role name>",\n'
        '      "techstack": [<list of technologies>],\n'
        '      "experience": "<required experience>",\n'
        '      "rationale": "<reasoning behind the recommendation>"\n'
        "    }},\n"
        "    ...\n"
        "  ]\n"
        "}}"
    )
    try:
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "feedback_text"])
        prompt_text = prompt.format(context=context, question=user_question, feedback_text=feedback_text)
        response = llm.invoke([HumanMessage(content=prompt_text)])
        return response
    except Exception as e:
        print(f"Error generating JSON response: {e}")
        return None

# --- New Tasks for Evaluator–Optimizer Approach ---
@task
def llm_generate_analysis_task(retrieved_chunks, user_question, feedback: str = None):
    return generate_json_response(retrieved_chunks, user_question, feedback)

@task
def llm_evaluation_task(response_text: str):
    """
    Evaluates the JSON response using the evaluator with structured output.
    Checks that the JSON adheres to the expected schema and validity,
    and also evaluates whether the response from the LLM sufficiently answers the question.
    """
    try:
        evaluation_prompt = (
            f"Evaluate the following JSON response. Verify that it adheres to the expected schema "
            f"and is valid JSON. Additionally, assess whether the answer provided by the LLM fully "
            f"addresses the user question with sufficient detail and clarity. Provide a grade of "
            f"'acceptable' or 'not acceptable' and include feedback for improvement if necessary:\n"
            f"{response_text}"
        )
        feedback_obj = evaluator.invoke([HumanMessage(content=evaluation_prompt)])
        return feedback_obj
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

# --- Existing Workflow Tasks ---
@task
def extract_text_task(folder_path: str):
    return process_folder(folder_path)

@task
def split_text_task(text: str):
    return split_text(text)

@task
def vector_store_task(chunks):
    return build_vector_store(chunks)

@task
def retrieve_chunks_task(vectorstore, query: str):
    return retrieve_relevant_chunks(vectorstore, query, k=5)

# --- Updated Main Workflow with Evaluator–Optimizer ---
@entrypoint()
def document_analysis_workflow(state: dict):
    folder_path = state["folder_path"]
    user_question = state["user_question"]
    # Extract project name from folder path (e.g., the last part of the path)
    project_name = os.path.basename(os.path.normpath(folder_path))
    
    # Step 1: Extract text from the folder
    try:
        full_text = extract_text_task(folder_path).result()
    except Exception as e:
        return {"error": f"Failed during text extraction: {e}"}
    
    # Step 2: Split text into manageable chunks
    try:
        chunks = split_text_task(full_text).result()
    except Exception as e:
        return {"error": f"Failed during text splitting: {e}"}
    
    # Step 3: Build a vector store from the chunks
    vectorstore = vector_store_task(chunks).result()
    if vectorstore is None:
        return {"error": "Vector store creation failed."}
    
    # Step 4: Retrieve the most relevant chunks based on the user question
    try:
        retrieved_chunks = retrieve_chunks_task(vectorstore, user_question).result()
    except Exception as e:
        return {"error": f"Failed during chunk retrieval: {e}"}
    
    # Evaluator-Optimizer loop with maximum iterations to avoid infinite loops
    feedback = None
    max_iterations = 5
    iteration = 0
    result_json = None

    while iteration < max_iterations:
        iteration += 1
        try:
            # Generate a JSON response (incorporating feedback if any)
            llm_response = llm_generate_analysis_task(retrieved_chunks, user_question, feedback).result()
            response_text = llm_response.content.strip()
        except Exception as e:
            feedback = f"LLM generation error: {e}"
            continue

        if response_text.startswith("```"):
            parts = response_text.split("```")
            if len(parts) >= 3:
                json_text = parts[1].strip()
                if json_text.lower().startswith("json"):
                    json_text = json_text[4:].strip()
                response_text = json_text

        try:
            # Evaluate the JSON response
            eval_feedback = llm_evaluation_task(response_text).result()
            eval_data = (
                eval_feedback.model_dump() if hasattr(eval_feedback, "model_dump") 
                else json.loads(eval_feedback.content)
            )
        except Exception as e:
            eval_data = {"grade": "not acceptable", "feedback": "Evaluation parsing error."}

        if eval_data.get("grade") == "acceptable":
            try:
                result_json = json.loads(response_text)
            except json.JSONDecodeError as e:
                result_json = {"error": "Invalid JSON response", "response": response_text}
            break
        else:
            # Use the evaluator's feedback to improve the next generation
            feedback = eval_data.get("feedback", "Please improve the response.")
            print(f"Iteration {iteration} feedback from evaluator: {feedback}")
    
    if result_json is None:
        return {"error": "Maximum iterations reached without acceptable response."}
    
    # Add the project name to the final JSON result
    return {"project_name": project_name, "result": result_json}

# --- Sample Invocation ---
if __name__ == "__main__":
    folder_path = "/Users/mikita/Projects/Agents/TeamRecommendations_Evaluator_Optimizer/Description/Zava"  
    user_question = (
        "Based on the project description extracted from the documents, "
        "please generate a JSON object with keys 'project_description' and 'team_recommendation'. "
        "The 'team_recommendation' should include the following fields for each recommendation: "
        "role, techstack, experience, and rationale."
    )
    input_state = {"folder_path": folder_path, "user_question": user_question}
    config = {"configurable": {"thread_id": "unique_thread_id"}}  # Provide a unique thread id

    # Invoke the workflow synchronously
    final_result = document_analysis_workflow.invoke(input_state, config=config)
    
    # Define the output folder path
    output_folder = "/Users/mikita/Projects/Agents/TeamRecommendations_Evaluator_Optimizer/Result"
    os.makedirs(output_folder, exist_ok=True)
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_file = os.path.join(output_folder, folder_name + ".json")

    # Save the final JSON result
    with open(output_file, "w") as f:
        json.dump(final_result, f, indent=2)

    print(f"Saved JSON result to {output_file}")
