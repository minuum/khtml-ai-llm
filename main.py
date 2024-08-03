from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import sys
from typing import List
from dotenv import load_dotenv
# 현재 스크립트의 디렉토리 경로를 파이썬 경로에 추가
# sys.path.append(os.pardir)
sys.path.append("/home/user/khtml-ai-llm/app")
import sys

print(sys.path)

from app.ai_assistant import AIAssistant
from app.store_db import DocumentProcessor
# 환경 변수 로드 (API 키 등)
load_dotenv()

# FastAPI 애플리케이션 초기화
app = FastAPI()

# OpenAI API 키와 벡터 DB 경로 설정
API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DB_PATH = "./vectordb"

# AI 어시스턴트 및 문서 처리기 초기화
ai_assistant = AIAssistant(api_key=API_KEY, vector_db_path=VECTOR_DB_PATH)
document_processor = DocumentProcessor(api_key=API_KEY, vector_db_path=VECTOR_DB_PATH)

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    """PDF 파일들을 업로드하고 벡터 DB에 저장합니다."""
    for file in files:
        # 임시 파일 경로 설정
        temp_file_path = os.path.join("/tmp", file.filename)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # PDF 파일을 벡터 DB에 저장
        try:
            document_processor.process_documents(temp_file_path, data_format="pdf")
        except Exception as e:
            return HTTPException(status_code=500, detail=str(e))
        finally:
            # 임시 파일 삭제
            os.remove(temp_file_path)

    return JSONResponse(content={"message": "Files processed and stored successfully."})

@app.post("/ask/")
async def ask_question(question: str):
    """사용자의 질문에 대해 AI 어시스턴트의 응답을 반환합니다."""
    try:
        response = ai_assistant.get_rag_response(question)
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={"response": response})

@app.get("/")
async def root():
    """기본 엔드포인트."""
    return {"message": "Welcome to the AI Assistant API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
