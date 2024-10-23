# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/api/v1/models")
async def list_models():
    # 테스트용 더미 데이터
    return [{"id": 1, "name": "test_model", "version": "1.0.0", "status": "Production"}]


# Debug용 예외 처리 추가
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"detail": str(exc)}, 500
