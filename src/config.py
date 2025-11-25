"""
중앙 설정 관리 모듈

config.yaml을 로드하고 모든 모듈에서 일관된 설정값을 사용할 수 있도록 합니다.
"""

from pathlib import Path
from typing import Any, Optional
import yaml


# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# config.yaml 경로
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# 캐시된 설정
_config_cache: Optional[dict] = None


def load_config(reload: bool = False) -> dict:
    """
    config.yaml 로드

    Args:
        reload: True면 캐시를 무시하고 다시 로드

    Returns:
        설정 딕셔너리
    """
    global _config_cache

    if _config_cache is not None and not reload:
        return _config_cache

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config_cache = yaml.safe_load(f)

    return _config_cache


def get(key: str, default: Any = None) -> Any:
    """
    설정값 조회 (dot notation 지원)

    Args:
        key: 설정 키 (예: "data.input_path", "unsupervised.min_topic_size")
        default: 키가 없을 때 반환할 기본값

    Returns:
        설정값

    Examples:
        >>> get("data.input_path")
        "data/input/articles_naver.json"
        >>> get("unsupervised.min_topic_size", 5)
        10
    """
    config = load_config()

    keys = key.split(".")
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


def get_input_path() -> str:
    """입력 데이터 경로 반환"""
    return get("data.input_path", "data/input/articles.json")


def get_processed_path() -> str:
    """처리된 데이터 경로 반환"""
    return get("data.processed_path", "data/processed/")


def get_results_dir() -> str:
    """결과 디렉토리 경로 반환"""
    return get("output.results_dir", "results/")


def get_models_dir() -> str:
    """모델 디렉토리 경로 반환"""
    return get("output.models_dir", "models/")


def get_figures_dir() -> str:
    """시각화 디렉토리 경로 반환"""
    return get("output.figures_dir", "results/figures/")


def get_media_outlets() -> list:
    """언론사명 리스트 반환"""
    return get("preprocessing.media_outlets", [])


def get_stopwords() -> list:
    """불용어 리스트 반환"""
    return get("preprocessing.stopwords", [])
