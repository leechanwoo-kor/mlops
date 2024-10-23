# app/services/mlflow.py
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MLflowService:
    def __init__(self):
        self.client = MlflowClient()
        
    async def create_experiment(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """새로운 실험 생성"""
        try:
            experiment = mlflow.get_experiment_by_name(name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(name, tags=tags)
                logger.info(f"Created new experiment: {name}")
                return experiment_id
            return experiment.experiment_id
        except Exception as e:
            logger.error(f"Error creating experiment: {str(e)}")
            raise

    async def log_model_training(
        self,
        experiment_name: str,
        model_name: str,
        params: Dict[str, any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[Dict[str, str]] = None
    ):
        """모델 학습 결과 기록"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = await self.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id

            with mlflow.start_run(experiment_id=experiment_id):
                # 파라미터 기록
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # 메트릭 기록
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                
                # 태그 기록
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)
                
                # 아티팩트 기록
                if artifacts:
                    for key, path in artifacts.items():
                        mlflow.log_artifact(path, key)
                
                # 모델 등록
                mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/model",
                    model_name
                )
                
            logger.info(f"Logged training results for model: {model_name}")
            return mlflow.active_run().info.run_id
        except Exception as e:
            logger.error(f"Error logging model training: {str(e)}")
            raise

    async def get_experiment_runs(
        self,
        experiment_name: str,
        filter_string: str = None,
        order_by: List[str] = None
    ):
        """실험의 실행 기록 조회"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment {experiment_name} not found")
                
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                order_by=order_by,
                run_view_type=ViewType.ACTIVE_ONLY
            )
            
            return [{
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(run.info.start_time/1000.0),
                "end_time": datetime.fromtimestamp(run.info.end_time/1000.0) if run.info.end_time else None,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            } for run in runs]
        except Exception as e:
            logger.error(f"Error getting experiment runs: {str(e)}")
            raise