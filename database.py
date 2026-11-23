import sqlite3               # 파이썬 기본 내장 경량 데이터베이스(SQLite3) 제어 모듈 (별도 설치 불필요)
from datetime import datetime  # 탐지 시점의 현재 날짜 및 시간을 기록하기 위한 표준 모듈


def init_db(db_path="deepfake_detector.db"):
    """
    [데이터베이스 초기화 함수]
    - 역할: 지정된 경로에 SQLite DB 파일(.db)을 생성(또는 연결)하고,
            탐지 로그를 저장할 'detection_logs' 테이블이 없으면 새로 생성합니다.
    - 매개변수:
        db_path (str): 생성하거나 연결할 데이터베이스 파일 경로 (기본값: "deepfake_detector.db")
    """
    # 1. SQLite 데이터베이스 파일에 연결 (해당 경로에 파일이 존재하지 않으면 자동으로 새 빈 DB 파일 생성)
    conn = sqlite3.connect(db_path)
    
    # 2. SQL 쿼리문을 실행하고 결과를 받아오기 위한 작업 도구(Cursor) 객체 생성
    cursor = conn.cursor()
    
    # 3. 테이블 생성 SQL 쿼리 실행
    #    - IF NOT EXISTS: 이미 테이블이 만들어져 있을 경우 에러를 내지 않고 안전하게 건너뜀
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 고유 번호: 데이터가 들어올 때마다 1씩 자동 증가하는 기본키(PK)
            filename TEXT NOT NULL,                -- 분석한 이미지 파일명 (문자열, 빈 값/NULL 허용 안 함)
            prediction TEXT NOT NULL,              -- AI 모델의 최종 판정 결과 (예: "REAL (진짜)" 또는 "FAKE (가짜)")
            fake_probability REAL NOT NULL,        -- 가짜일 확률 수치 (실수형 소수점 데이터, 예: 94.25)
            created_at TEXT NOT NULL               -- 탐지가 수행된 일시 문자열 (예: "2026-11-24 08:45:00")
        )
    ''')
    
    # 4. 변경 사항(테이블 생성 구조)을 실제 디스크의 .db 파일에 영구 반영(트랜잭션 확정)
    conn.commit()
    
    # 5. 데이터베이스 연결 종료 및 파일 잠금(Lock) 해제 (메모리 누수 및 동시 접근 충돌 방지)
    conn.close()
    print("[DB] SQLite 데이터베이스 및 테이블 준비 완료!")


def save_result_to_db(filename, prediction, fake_probability, db_path="deepfake_detector.db"):
    """
    [탐지 결과 저장 함수]
    - 역할: AI 모델이 판별한 단일 이미지의 분석 결과를 'detection_logs' 테이블에 새로운 행(Row)으로 삽입합니다.
    - 매개변수:
        filename (str): 업로드/검사된 이미지 파일 이름
        prediction (str): 최종 판별 라벨 ("REAL (진짜)" / "FAKE (가짜)")
        fake_probability (float): 가짜로 예측된 확률 백분율 (0.0 ~ 100.0)
        db_path (str): 대상 데이터베이스 파일 경로
    """
    # 1. 기존 데이터베이스 파일에 연결
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 2. 함수가 호출된 정확한 현재 시각을 "YYYY-MM-DD HH:MM:SS" 형식의 문자열로 추출
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 3. 데이터 삽입(INSERT) SQL 쿼리 실행
    #    - 보안 포인트(Parameterized Query): 문자열 포맷팅(f-string) 대신 '?' 플레이스홀더와 튜플을 사용하여
    #      파일명에 특수문자(' 등)가 포함되거나 SQL 인젝션(SQL Injection) 공격이 들어와도 안전하게 처리됨
    #    - id 컬럼은 AUTOINCREMENT 속성이므로 값을 넣지 않아도 DB가 자동으로 번호를 부여함
    cursor.execute('''
        INSERT INTO detection_logs (filename, prediction, fake_probability, created_at)
        VALUES (?, ?, ?, ?)
    ''', (filename, prediction, fake_probability, current_time))
    
    # 4. 삽입된 레코드(행)를 실제 .db 파일에 영구 저장
    conn.commit()
    
    # 5. 데이터베이스 연결 안전 종료
    conn.close()
    print(f"[DB 저장 완료] 파일: {filename} | 판정: {prediction} | 가짜 확률: {fake_probability:.2f}%")


# ==============================================================================
# [단독 실행 테스트 블록] 이 파일(database.py)을 직접 실행할 때만 작동하는 검증 코드
# ==============================================================================
if __name__ == "__main__":
    # 1. 테이블 초기화 테스트
    init_db()
    
    # 2. 가상의 탐지 결과 1건 저장 테스트
    save_result_to_db(
        filename="test_sample.jpg",
        prediction="FAKE (가짜)",
        fake_probability=92.45
    )
