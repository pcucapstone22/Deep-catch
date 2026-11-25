import subprocess
import sys

# ========================================================
# [설정] pip 설치 패키지명과 파이썬 내부 import 이름 매핑
# - key: pip install에 사용하는 라이브러리명
# - value: 파이썬 코드에서 import 할 때 사용하는 모듈명
# ========================================================
REQUIRED_PACKAGES = {
    "torch": "torch",                   # PyTorch 딥러닝 프레임워크
    "torchvision": "torchvision",       # 컴퓨터 비전 모델 및 전처리 도구
    "opencv-python": "cv2",             # 이미지 처리 및 DCT 주파수 변환
    "numpy": "numpy",                   # 다차원 배열 및 수치 연산
    "scipy": "scipy",                   # 2D DCT 주파수 변환 연산
    "Pillow": "PIL",                    # 이미지 파일 입출력 (import PIL)
    "streamlit": "streamlit",           # 웹 대시보드 및 UI 구현
    "pandas": "pandas",                 # 데이터프레임 및 DB 로그 표 처리
    "fastapi": "fastapi",               # FastAPI 백엔드 웹 프레임워크
    "uvicorn": "uvicorn",               # FastAPI 구동용 ASGI 웹 서버
    "python-multipart": "multipart"     # FastAPI 이미지 파일 업로드(UploadFile) 처리
}

def install_and_verify():
    """
    필수 라이브러리의 설치 여부를 전수 조사하고, 
    누락된 라이브러리만 자동으로 pip 설치를 수행하는 함수
    """
    print("=" * 60)
    print("[환경 점검] 딥페이크 탐지 프로젝트 필수 라이브러리 검사 시작")
    print("=" * 60)
    
    # 1. pip 자체를 최신 버전으로 먼저 업데이트 (설치 오류 방지)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    except Exception as e:
        print(f"[경고] pip 업데이트 중 경고 발생: {e}")

    installed_count = 0
    missing_packages = []

    # 2. 패키지 목록을 순회하며 설치 확인
    for pip_name, import_name in REQUIRED_PACKAGES.items():
        try:
            # 동적으로 모듈 import 시도
            __import__(import_name)
            print(f"  [✓] {pip_name:<18} : 이미 설치되어 있습니다.")
            installed_count += 1
        except ImportError:
            print(f"  [X] {pip_name:<18} : 미설치 상태입니다. (설치 대상)")
            missing_packages.append(pip_name)

    # 3. 누락된 패키지가 있을 경우에만 자동 설치 진행
    if missing_packages:
        print("\n[자동 설치 시작] 누락된 패키지들을 순차적으로 설치합니다...")
        for pkg in missing_packages:
            print(f"-> {pkg} 설치 중...")
            try:
                # 현재 사용 중인 파이썬 실행 파일(sys.executable) 경로의 pip로 설치
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"  └─ {pkg} 설치 성공!")
            except subprocess.CalledProcessError as e:
                print(f"  └─ [오류] {pkg} 설치 실패! 에러 코드: {e}")
    else:
        print("\n[완료] 모든 필수 라이브러리가 이미 완벽하게 준비되어 있습니다!")

    print("=" * 60)


# ========================================================
# 독립 실행 스위치: 이 파일(setup_env.py)을 직접 실행할 때 작동
# ========================================================
if __name__ == "__main__":
    install_and_verify()
