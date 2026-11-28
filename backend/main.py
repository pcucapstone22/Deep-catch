import io
import os
import sqlite3
import numpy as np
import scipy.fftpack
import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from torchvision import models, transforms

# ==============================================================================
# [0] 추론 정합성 핵심 설정 스위치 (학습 환경과 일치시켜야 하는 항목) [각주 1]
# ==============================================================================
# 1) 클래스 인덱스 설정:
#    - 커스텀 DualStreamDataset 기준: {'real': 0, 'fake': 1} -> FAKE_CLASS_INDEX = 1
#    - torchvision ImageFolder 알파벳 순 기준: {'fake': 0, 'real': 1} -> FAKE_CLASS_INDEX = 0
FAKE_CLASS_INDEX = 1

# 2) 2D DCT 인스턴스 정규화 적용 여부:
#    - 기존 학습된 .pth 파일의 입력 분포(정규화 미적용 log(|DCT| + 1e-5))와 일치시키기 위해 False로 설정
#    - True로 설정할 경우 학습 때와 다른 스케일 텐서가 입력되어 Softmax 출력 포화(100% 고정) 발생 가능
USE_DCT_INSTANCE_NORM = False

# 3) 엄격한 가중치 키 검증 여부:
#    - True 권장: 학습 모델과 레이어 변수명이 1개라도 다르면 조용히 무시하지 않고 즉시 오류 출력
STRICT_WEIGHT_LOADING = True


# ==============================================================================
# [1] 경로 및 SQLite 데이터베이스 모듈 [각주 2]
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # 현재 main.py가 실행되는 절대 경로
PARENT_DIR = os.path.dirname(BASE_DIR)                     # 프로젝트 루트 경로 (backend/와 frontend/ 분리 구조 탐색용)
DB_PATH = os.path.join(BASE_DIR, "deepfake_detector.db")   # 탐지 이력 영구 저장용 SQLite DB 파일 경로
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")         # 학습 완료된 .pth 가중치 파일 기본 디렉토리


def _resolve_frontend_dir() -> str:
    """
    index.html, style.css, app.js가 위치한 프론트엔드 폴더를 자동 탐색합니다.
    - 백엔드와 프론트엔드가 같은 폴더에 있는 환경과 분리된 폴더 구조를 모두 지원합니다.
    """
    frontend_sibling = os.path.join(PARENT_DIR, "frontend")
    for candidate in (BASE_DIR, frontend_sibling, PARENT_DIR):
        if os.path.exists(os.path.join(candidate, "index.html")):
            return candidate
    return BASE_DIR


FRONTEND_DIR = _resolve_frontend_dir()


def init_db(db_path: str = DB_PATH) -> None:
    """
    탐지 이력을 저장할 SQLite 테이블(detection_logs)을 생성합니다.
    - IF NOT EXISTS 구문으로 서버 재시작 시에도 기존 탐지 기록이 초기화되지 않도록 보호합니다.
    """
    abs_path = os.path.abspath(db_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    conn = sqlite3.connect(abs_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                fake_probability REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        # 예외 발생 여부와 관계없이 SQLite 파일 잠금(Database Lock) 방지를 위해 연결 종료
        conn.close()


def save_result_to_db(
    filename: str,
    prediction: str,
    fake_probability: float,
    db_path: str = DB_PATH,
) -> None:
    """
    판별 결과를 detection_logs 테이블에 기록합니다.
    - SQL 인젝션 공격을 방지하기 위해 문자열 포매팅 대신 파라미터 바인딩(?, ?, ?)을 사용합니다.
    """
    abs_path = os.path.abspath(db_path)
    init_db(abs_path)

    conn = sqlite
