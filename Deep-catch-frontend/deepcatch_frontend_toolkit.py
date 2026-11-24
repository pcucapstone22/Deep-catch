# -*- coding: utf-8 -*-
"""
================================================================================
Deep-catch 프론트엔드 구동·수정·연동 통합 클래스 모음 (deepcatch_frontend_toolkit.py)
작성: 송동화 (프론트엔드 UI/UX 및 연동 가이드)
================================================================================

[이 파일의 목적]
누구나(프론트엔드·백엔드·AI 팀원 모두) 이 파일 하나만 열면 다음 작업을 즉시 수행할 수 있도록
핵심 기능을 **역할별 파이썬 클래스(Class)**로 모듈화해 정리한 실행 및 수정 도구입니다.

  1. FrontendConfigManager : 프론트엔드(`index.html`)의 외부 도메인 백엔드 URL을
                             코드로 손쉽게 수정·고정·초기화하는 설정 관리 클래스
  2. StaticFrontendServer  : 별도 설치 없이 내 PC에서 프론트엔드 웹 화면(`5500` 포트)을
                             즉시 실행하는 경량 HTTP 서버 클래스
  3. DetectionLogRepository: SQLite(`deepfake_detector.db`)의 `detection_logs` 테이블을
                             생성·저장·조회하는 데이터베이스 관리 클래스
  4. DeepfakeInferenceBridge: 백엔드 AI 모델(`ensemble_detector.py` + `.pth`)이 준비된 경우
                             실제 딥페이크 판별을 수행하고, 아직 준비 전이라면 테스트용
                             시뮬레이션(Mock) 추론을 자동 대체 실행하는 브릿지 클래스
  5. DeepcatchAPIServer    : 프론트엔드(`app.js`)의 `/api/predict`, `/api/logs` 요청을
                             외부 도메인(CORS 허용)에서도 안전하게 받아 처리하는 FastAPI 서버 클래스

[빠른 실행 방법]
  터미널에서 아래 명령어를 실행하면 프론트엔드(5500) + 백엔드 API(8000)가 한 번에 켜집니다.
  $ pip install fastapi uvicorn python-multipart
  $ python deepcatch_frontend_toolkit.py --mode all
================================================================================
"""

import os
import re
import sys
import time
import random
import sqlite3
import threading
import argparse
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


# ==============================================================================
# [클래스 1] 프론트엔드 설정 및 외부 도메인 주입 관리 클래스
# ==============================================================================
class FrontendConfigManager:
    """
    [역할]
    frontend/index.html 파일을 직접 열어 수정하지 않고도,
    외부 백엔드 도메인(예: https://api.deep-catch.site 또는 ngrok 주소)을
    HTML 전역 변수(`window.DEEP_CATCH_API_URL`)로 안전하게 삽입하거나 제거하는 클래스입니다.

    [언제 사용하나요?]
    - GitHub Pages / Vercel / Netlify 등 외부 클라우드에 프론트엔드를 배포하기 직전,
      백엔드 서버의 고정 도메인을 코드에 미리 박아넣고 싶을 때 사용합니다.
    """

    INJECT_START_MARKER = "<!-- [DEEPCATCH_API_CONFIG_START] -->"
    INJECT_END_MARKER = "<!-- [DEEPCATCH_API_CONFIG_END] -->"

    def __init__(self, frontend_dir: str = "./frontend"):
        self.frontend_dir = os.path.abspath(frontend_dir)
        self.index_path = os.path.join(self.frontend_dir, "index.html")

    def validate_files(self) -> bool:
        """필수 프론트엔드 파일 3종(index.html, style.css, app.js) 존재 여부를 점검합니다."""
        required = ["index.html", "style.css", "app.js"]
        missing = [f for f in required if not os.path.exists(os.path.join(self.frontend_dir, f))]
        if missing:
            print(f"[경고] '{self.frontend_dir}' 경로에 누락된 파일이 있습니다: {missing}")
            return False
        print(f"[확인 완료] 프론트엔드 핵심 파일 3종 정상 확인 ({self.frontend_dir})")
        return True

    def set_backend_domain(self, backend_url: str) -> None:
        """
        index.html 내부에 외부 백엔드 도메인 설정 스크립트를 자동 삽입(또는 갱신)합니다.
        :param backend_url: 연결할 백엔드 서버의 외부 도메인 (예: 'https://api.deep-catch.site')
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"index.html을 찾을 수 없습니다: {self.index_path}")

        clean_url = backend_url.strip().rstrip("/")
        if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
            raise ValueError("도메인 주소는 반드시 'http://' 또는 'https://'로 시작해야 합니다.")

        with open(self.index_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # 기존에 주입된 설정 블록이 있다면 먼저 제거
        html_content = self._strip_existing_block(html_content)

        # 새로 삽입할 설정 스크립트 블록 구성
        inject_block = (
            f"  {self.INJECT_START_MARKER}\n"
            f"  <script>\n"
            f'    window.DEEP_CATCH_API_URL = "{clean_url}";\n'
            f"  </script>\n"
            f"  {self.INJECT_END_MARKER}\n"
        )

        # <script src="app.js" defer></script> 바로 위에 삽입
        target_tag = '<script src="app.js" defer></script>'
        if target_tag not in html_content:
            raise RuntimeError('index.html에서 \'<script src="app.js" defer></script>\' 태그를 찾지 못했습니다.')

        updated_html = html_content.replace(target_tag, f"{inject_block}  {target_tag}")
        with open(self.index_path, "w", encoding="utf-8") as f:
            f.write(updated_html)

        print(f"[도메인 설정 완료] 프론트엔드 기본 백엔드 주소가 '{clean_url}'로 고정되었습니다.")

    def clear_backend_domain(self) -> None:
        """index.html에 고정해 둔 외부 백엔드 도메인 설정을 제거하고 자동 감지 모드로 되돌립니다."""
        if not os.path.exists(self.index_path):
            return

        with open(self.index_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        updated_html = self._strip_existing_block(html_content)
        with open(self.index_path, "w", encoding="utf-8") as f:
            f.write(updated_html)

        print("[도메인 초기화 완료] index.html의 고정 도메인 설정이 제거되었습니다.")

    def _strip_existing_block(self, html: str) -> str:
        pattern = re.compile(
            rf"\s*{re.escape(self.INJECT_START_MARKER)}.*?{re.escape(self.INJECT_END_MARKER)}\s*",
            re.DOTALL,
        )
        return re.sub(pattern, "\n", html)


# ==============================================================================
# [클래스 2] 프론트엔드 정적 웹 서버 실행 클래스 (Live Server 대체용)
# ==============================================================================
class StaticFrontendServer:
    """
    [역할]
    VS Code Live Server가 없어도 파이썬 표준 라이브러리만으로
    `frontend/` 폴더를 웹 서버(기본 5500번 포트)로 띄워주는 클래스입니다.

    [주요 특징]
    - 브라우저 캐시로 인해 수정한 CSS/JS가 안 바뀌는 현상을 막기 위해
      `Cache-Control: no-store` 헤더를 자동으로 응답에 실어줍니다.
    """

    def __init__(self, frontend_dir: str = "./frontend", host: str = "0.0.0.0", port: int = 5500):
        self.frontend_dir = os.path.abspath(frontend_dir)
        self.host = host
        self.port = port
        self._httpd: Optional[ThreadingHTTPServer] = None

    def start(self, blocking: bool = True) -> None:
        """
        프론트엔드 웹 서버를 시작합니다.
        :param blocking: True면 현재 터미널을 점유하며 대기, False면 백그라운드 스레드로 실행
        """
        directory = self.frontend_dir

        class NoCacheRequestHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

            def end_headers(self):
                # 개발 및 수정 시 브라우저 캐시로 인한 구버전 파일 로딩 방지
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                super().end_headers()

            def log_message(self, format, *args):
                # 콘솔 로그를 깔끔하게 정리해 출력
                print(f"[Frontend HTTP] {self.address_string()} - {format % args}")

        self._httpd = ThreadingHTTPServer((self.host, self.port), NoCacheRequestHandler)
        print(f"[프론트엔드 서버 실행] http://localhost:{self.port} (서빙 경로: {self.frontend_dir})")

        if blocking:
            try:
                self._httpd.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
            thread.start()

    def stop(self) -> None:
        """실행 중인 프론트엔드 웹 서버를 안전하게 종료합니다."""
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            print("[프론트엔드 서버 종료]")


# ==============================================================================
# [클래스 3] SQLite 탐지 이력 데이터베이스 관리 클래스
# ==============================================================================
class DetectionLogRepository:
    """
    [역할]
    기존 백엔드의 `database.py` 스키마(`deepfake_detector.db` -> `detection_logs` 테이블)와
    100% 동일한 구조로 탐지 기록을 저장하고, 프론트엔드 테이블(`/api/logs`)이 요구하는
    JSON 리스트 형태로 최신순 조회를 수행하는 클래스입니다.
    """

    def __init__(self, db_path: str = "deepfake_detector.db"):
        self.db_path = db_path
        self.init_table()

    def init_table(self) -> None:
        """detection_logs 테이블이 없으면 자동 생성합니다."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detection_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    fake_probability REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def insert_log(self, filename: str, prediction: str, fake_probability: float) -> Dict[str, Any]:
        """새로운 판별 결과를 DB에 저장하고 저장된 레코드 딕셔너리를 반환합니다."""
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO detection_logs (filename, prediction, fake_probability, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (filename, prediction, round(float(fake_probability), 2), created_at),
            )
            row_id = cursor.lastrowid
            conn.commit()

        return {
            "id": row_id,
            "filename": filename,
            "prediction": prediction,
            "fake_probability": round(float(fake_probability), 2),
            "created_at": created_at,
        }

    def fetch_recent_logs(self, limit: int = 30) -> List[Dict[str, Any]]:
        """최근 탐지 이력을 최신순(id DESC)으로 조회하여 JSON 직렬화 가능한 리스트로 반환합니다."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, filename, prediction, fake_probability, created_at
                FROM detection_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


# ==============================================================================
# [클래스 4] AI 딥페이크 추론 브릿지 클래스 (실제 모델 / Mock 테스트 자동 전환)
# ==============================================================================
class DeepfakeInferenceBridge:
    """
    [역할]
    프론트엔드에서 업로드된 이미지 바이트를 받아 3종 앙상블 모델(Spatial, Frequency, DualStream)로
    판별 결과를 산출하는 클래스입니다.

    [자동 전환(Fallback) 기능]
    - 같은 폴더에 `ensemble_detector.py`, `composite_detector_models.py`, `dataset.py`와
      학습된 가중치 파일 3개(`best_Spatial.pth`, `best_Frequency.pth`, `best_DualStream.pth`)가
      모두 있으면 **실제 PyTorch AI 모델**로 추론합니다.
    - 아직 학습된 `.pth` 파일이 없는 개발 단계라면 에러를 내지 않고 자동으로
      **시뮬레이션(Mock) 추론 모드**로 동작하여 프론트엔드 UI와 DB 연동을 테스트할 수 있게 해줍니다.
    """

    def __init__(
        self,
        spatial_weight: str = "best_Spatial.pth",
        freq_weight: str = "best_Frequency.pth",
        dual_weight: str = "best_DualStream.pth",
    ):
        self.is_real_model_loaded = False
        self.detector = None
        self.device = None
        self.spatial_transform = None
        self.generate_dct_fn = None

        self._try_load_real_models(spatial_weight, freq_weight, dual_weight)

    def _try_load_real_models(self, spatial_w: str, freq_w: str, dual_w: str) -> None:
        """실제 PyTorch 모듈과 .pth 가중치가 존재할 경우에만 모델을 메모리에 적재합니다."""
        weights_exist = all(os.path.exists(p) for p in [spatial_w, freq_w, dual_w])
        if not weights_exist:
            print("[추론 엔진 안내] .pth 가중치 파일이 없어 [시뮬레이션(Mock) 테스트 모드]로 동작합니다.")
            return

        try:
            import torch
            from torchvision import transforms
            from dataset import generate_dct_frequency_map
            from ensemble_detector import EnsembleDeepfakeDetector

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.detector = EnsembleDeepfakeDetector(
                spatial_path=spatial_w,
                freq_path=freq_w,
                dual_path=dual_w,
                device=self.device,
            )
            self.spatial_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.generate_dct_fn = generate_dct_frequency_map
            self.is_real_model_loaded = True
            print(f"[추론 엔진 안내] 실제 PyTorch 3종 앙상블 모델 로드 성공! (장치: {self.device})")
        except Exception as exc:
            print(f"[추론 엔진 경고] 실제 모델 로드 중 오류 발생 -> 시뮬레이션 모드로 전환됩니다: {exc}")
            self.is_real_model_loaded = False

    def predict_image_bytes(self, filename: str, image_bytes: bytes) -> Dict[str, Any]:
        """이미지 바이너리를 입력받아 프론트엔드 규격의 JSON 딕셔너리를 반환합니다."""
        if self.is_real_model_loaded:
            return self._predict_with_pytorch(filename, image_bytes)
        return self._predict_with_mock(filename)

    def _predict_with_pytorch(self, filename: str, image_bytes: bytes) -> Dict[str, Any]:
        """실제 SpatialCNN + FrequencyNoise + DualStream 모델로 추론을 수행합니다."""
        import io
        import numpy as np
        from PIL import Image

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb_tensor = self.spatial_transform(pil_img).unsqueeze(0)  # [1, 3, 256, 256]
        dct_tensor = self.generate_dct_fn(np.array(pil_img), target_size=(256, 256)).unsqueeze(0)  # [1, 1, 256, 256]

        final_pred, ensemble_prob = self.detector.predict_and_ensemble(
            filename=filename,
            rgb_tensor=rgb_tensor,
            dct_tensor=dct_tensor,
        )
        is_fake = str(final_pred).upper() == "FAKE" or ensemble_prob >= 50.0
        label = "FAKE (가짜)" if is_fake else "REAL (진짜)"

        return {
            "filename": filename,
            "prediction": label,
            "is_fake": is_fake,
            "fake_probability": round(float(ensemble_prob), 2),
            "mode": "pytorch_ensemble",
        }

    def _predict_with_mock(self, filename: str) -> Dict[str, Any]:
        """가중치(.pth)가 없는 환경에서 UI/UX 및 DB 연동을 검증하기 위한 시뮬레이션 추론입니다."""
        time.sleep(0.4)  # 실제 추론 대기 UX(로딩 상태) 확인용 짧은 지연
        s_prob = round(random.uniform(12.0, 94.0), 2)
        f_prob = round(random.uniform(12.0, 94.0), 2)
        d_prob = round(random.uniform(12.0, 94.0), 2)

        # README 및 ensemble_detector.py와 동일한 가중 평균 비율(0.3 : 0.3 : 0.4) 적용
        ensemble_prob = round(s_prob * 0.3 + f_prob * 0.3 + d_prob * 0.4, 2)
        is_fake = ensemble_prob >= 50.0
        label = "FAKE (가짜)" if is_fake else "REAL (진짜)"

        return {
            "filename": filename,
            "prediction": label,
            "is_fake": is_fake,
            "fake_probability": ensemble_prob,
            "spatial_probability": s_prob,
            "frequency_probability": f_prob,
            "dual_probability": d_prob,
            "mode": "mock_simulation",
        }


# ==============================================================================
# [클래스 5] FastAPI 백엔드 API 서버 클래스 (외부 도메인 CORS 완벽 지원)
# ==============================================================================
class DeepcatchAPIServer:
    """
    [역할]
    프론트엔드(`app.js`)가 호출하는 REST API 엔드포인트를 제공하는 백엔드 서버 클래스입니다.
      - GET  /api/health  : 외부 도메인에서 서버 생존 여부 확인
      - GET  /api/logs    : SQLite DB의 최근 탐지 이력 조회
      - POST /api/predict : 업로드 이미지 분석 + DB 자동 저장 + 결과 JSON 반환
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        db_path: str = "deepfake_detector.db",
        allowed_origins: Optional[List[str]] = None,
    ):
        self.host = host
        self.port = port
        self.repo = DetectionLogRepository(db_path=db_path)
        self.bridge = DeepfakeInferenceBridge()
        self.allowed_origins = allowed_origins or ["*"]
        self.app = self._build_fastapi_app()

    def _build_fastapi_app(self):
        from fastapi import FastAPI, File, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI(
            title="Deep-catch API Server",
            description="공간·주파수 듀얼 스트림 앙상블 딥페이크 탐지 백엔드 API",
            version="1.0.0",
        )

        # 외부 도메인(GitHub Pages, Vercel, Cloudflare Tunnel 등)에서의 접속을 허용하는 CORS 설정
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.allowed_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/api/health")
        def health_check():
            return {
                "status": "ok",
                "real_model_loaded": self.bridge.is_real_model_loaded,
                "timestamp": datetime.now().isoformat(),
            }

        @app.get("/api/logs")
        def get_detection_logs(limit: int = 30):
            return self.repo.fetch_recent_logs(limit=limit)

        @app.post("/api/predict")
        async def predict_deepfake(file: UploadFile = File(...)):
            if not file.filename:
                raise HTTPException(status_code=400, detail="업로드된 파일명이 비어 있습니다.")

            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="빈 이미지 파일은 분석할 수 없습니다.")

            # 1. 추론 실행 (실제 PyTorch 모델 또는 Mock 시뮬레이션)
            result = self.bridge.predict_image_bytes(filename=file.filename, image_bytes=content)

            # 2. 실제 PyTorch 모델은 내부에서 DB 저장을 수행하므로, Mock 모드일 때만 여기서 DB 기록
            if result.get("mode") == "mock_simulation":
                self.repo.insert_log(
                    filename=result["filename"],
                    prediction=result["prediction"],
                    fake_probability=result["fake_probability"],
                )

            return result

        return app

    def run(self) -> None:
        """Uvicorn ASGI 서버로 FastAPI 앱을 실행합니다."""
        import uvicorn

        print(f"[백엔드 API 서버 실행] http://localhost:{self.port} (외부 도메인 CORS 허용됨)")
        uvicorn.run(self.app, host=self.host, port=self.port)


# ==============================================================================
# [실행 진입점] 터미널 명령어 옵션에 따라 원하는 기능만 골라서 즉시 실행
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep-catch 프론트엔드 구동·수정·연동 통합 도구")
    parser.add_argument(
        "--mode",
        choices=["all", "frontend", "backend", "set-domain", "reset-domain"],
        default="all",
        help="실행 모드 선택 (all: 프론트+백엔드 동시 실행, frontend: 웹화면만, backend: API만, set-domain: 외부 백엔드 도메인 고정)",
    )
    parser.add_argument("--frontend-dir", default="./frontend", help="프론트엔드 폴더 경로 (기본값: ./frontend)")
    parser.add_argument("--frontend-port", type=int, default=5500, help="프론트엔드 포트 번호 (기본값: 5500)")
    parser.add_argument("--backend-port", type=int, default=8000, help="백엔드 API 포트 번호 (기본값: 8000)")
    parser.add_argument("--api-url", default="", help="--mode set-domain 사용 시 고정할 외부 백엔드 도메인 URL")

    args = parser.parse_args()
    config_mgr = FrontendConfigManager(frontend_dir=args.frontend_dir)

    if args.mode == "set-domain":
        if not args.api_url:
            print("[오류] --api-url 옵션에 외부 백엔드 주소를 함께 입력하세요. 예: --api-url https://api.deep-catch.site")
            sys.exit(1)
        config_mgr.set_backend_domain(args.api_url)

    elif args.mode == "reset-domain":
        config_mgr.clear_backend_domain()

    elif args.mode == "frontend":
        config_mgr.validate_files()
        StaticFrontendServer(frontend_dir=args.frontend_dir, port=args.frontend_port).start(blocking=True)

    elif args.mode == "backend":
        DeepcatchAPIServer(port=args.backend_port).run()

    elif args.mode == "all":
        config_mgr.validate_files()
        # 프론트엔드(5500)는 백그라운드 스레드로 띄우고, 메인 스레드에서 백엔드 API(8000) 실행
        StaticFrontendServer(frontend_dir=args.frontend_dir, port=args.frontend_port).start(blocking=False)
        DeepcatchAPIServer(port=args.backend_port).run()
