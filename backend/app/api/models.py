# backend/app/api/models.py
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ConfigDict
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class ModelTrainingRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    experiment_name: str
    model_name: str
    params: Dict[str, Any]  # any 대신 Any 사용
    metrics: Dict[str, float]
    tags: Optional[Dict[str, str]] = None
    artifacts: Optional[Dict[str, str]] = None

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@router.post("/train")
async def log_training(request: ModelTrainingRequest):
    """모델 학습 결과 기록"""
    logger.info(f"Received training request for model: {request.model_name}")
    try:
        from app.services.mlflow import MLflowService
        mlflow_service = MLflowService()
        run_id = await mlflow_service.log_model_training(
            experiment_name=request.experiment_name,
            model_name=request.model_name,
            params=request.params,
            metrics=request.metrics,
            tags=request.tags,
            artifacts=request.artifacts
        )
        logger.info(f"Successfully logged training with run_id: {run_id}")
        return {"run_id": run_id, "status": "success"}
    except Exception as e:
        logger.error(f"Error logging training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_name}/runs")
async def get_experiment_runs(
    experiment_name: str,
    filter_string: Optional[str] = None,
    order_by: Optional[List[str]] = None
):
    """실험의 실행 기록 조회"""
    logger.info(f"Fetching runs for experiment: {experiment_name}")
    try:
        from app.services.mlflow import MLflowService
        mlflow_service = MLflowService()
        runs = await mlflow_service.get_experiment_runs(
            experiment_name=experiment_name,
            filter_string=filter_string,
            order_by=order_by
        )
        return {"runs": runs}
    except ValueError as e:
        logger.error(f"Experiment not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting runs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))