from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.store_db import load_and_index_documents
from app.rag_chain import get_rag_response

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

# Initialize retriever
retriever = load_and_index_documents("./data/academic resources")

@app.post("/query/")
async def query_rag(request: QueryRequest):
    try:
        response = get_rag_response(retriever, request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
