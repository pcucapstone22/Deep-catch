import os
import tarfile
import gzip
import zipfile  # [.zip 압축 해제용] 파이썬 내장 표준 라이브러리 (별도 pip 설치 불필요)
import shutil

def extract_archives_in_dataset_dir(dataset_dir="./dataset", target_dir="./extracted_data"):
    """
    지정된 'dataset' 디렉토리 내를 스캔하여 
    존재하는 모든 .zip, .tar.gz, .tgz, .gz 형태의 압축 파일을 자동 감지해 해제하는 함수
    
    :param dataset_dir: 압축 파일들이 위치한 입력 폴더 경로 (기본값: ./dataset)
    :param target_dir: 압축 해제된 결과 파일들이 저장될 출력 폴더 경로 (기본값: ./extracted_data)
    """
    # =========================================================================
    # [1단계] 입력 디렉토리('dataset') 존재 여부 검사 및 자동 생성
    # =========================================================================
    # os.path.exists(): 지정한 경로에 파일이나 폴더가 존재하는지 확인 (True/False 반환)
    if not os.path.exists(dataset_dir):
        # os.makedirs(): 폴더를 생성. exist_ok=True 옵션은 이미 폴더가 있어도 에러를 내지 않음
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"[{dataset_dir}] 디렉토리가 존재하지 않아 새로 생성했습니다.")
        print(f"-> 압축을 풀 파일(.zip, .tar.gz, .gz 등)을 '{os.path.abspath(dataset_dir)}' 폴더에 넣고 다시 실행하세요.")
        return  # 파일이 없으므로 안내문 출력 후 함수 조기 종료

    # =========================================================================
    # [2단계] 'dataset' 폴더 내 모든 파일 탐색 및 지원 압축 확장자 필터링
    # =========================================================================
    # os.listdir(): 해당 폴더 안에 있는 모든 파일과 하위 폴더 이름들의 리스트를 반환
    all_files = os.listdir(dataset_dir)
    
    # 리스트 컴프리헨션(List Comprehension)을 사용해 지원하는 4가지 확장자로 끝나는 파일만 수집
    # endswith(): 문자열이 특정 튜플 안의 확장자로 끝나는지 검사
    archive_files = [f for f in all_files if f.endswith(('.zip', '.tar.gz', '.tgz', '.gz'))]

    # 조건에 맞는 압축 파일이 하나도 없는 경우 예외 처리
    if not archive_files:
        print(f"[{dataset_dir}] 폴더 내에 지원하는 압축 파일(.zip, .tar.gz, .gz)이 없습니다.")
        print(f"현재 폴더에 있는 파일 목록: {all_files}")
        return

    print(f"[{dataset_dir}] 디렉토리에서 총 {len(archive_files)}개의 압축 파일을 발견했습니다.")
    
    # 압축이 풀린 결과물이 저장될 target_dir 폴더가 없다면 사전 생성
    os.makedirs(target_dir, exist_ok=True)

    # =========================================================================
    # [3단계] 발견된 압축 파일들을 하나씩 순회하며 확장자별 분기 해제 실행
    # =========================================================================
    for file_name in archive_files:
        # os.path.join(): 운영체제(Windows/Mac/Linux)에 맞게 경로 구분자(/ 또는 \)를 자동으로 맞춰 결합
        # 예: ./dataset + my_data.zip -> ./dataset/my_data.zip
        file_path = os.path.join(dataset_dir, file_name)
        
        # ---------------------------------------------------------------------
        # [Case A] .zip 압축 파일 처리 구문 (새로 추가된 핵심 로직)
        # ---------------------------------------------------------------------
        if file_name.endswith('.zip'):
            print(f"\n▶ [{file_name}] .zip 파일 압축 해제 중...")
            
            # zipfile.ZipFile(): .zip 파일을 읽기 전용('r') 컨텍스트 객체로 열기
            # with 구문을 사용하여 작업 완료 후 파일 핸들을 자동으로 안전하게 닫아줌(close)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # zip_ref.extractall(): zip 파일 내의 모든 폴더 구조와 파일들을 target_dir 경로에 전부 풀기
                zip_ref.extractall(target_dir)
                
            print(f"  └─ [완료] '{os.path.abspath(target_dir)}' 폴더에 저장되었습니다.")

        # ---------------------------------------------------------------------
        # [Case B] .tar.gz 또는 .tgz 압축 파일 처리 구문 (여러 파일/폴더 아카이브)
        # ---------------------------------------------------------------------
        elif file_name.endswith(('.tar.gz', '.tgz')):
            print(f"\n▶ [{file_name}] .tar.gz 아카이브 파일 압축 해제 중...")
            
            # tarfile.open(): 'r:gz' 옵션으로 gzip 압축이 적용된 tar 묶음 파일을 염
            with tarfile.open(file_path, "r:gz") as tar:
                # tar.extractall(): 내부 폴더 경로와 파일들을 그대로 유지하면서 target_dir에 추출
                tar.extractall(path=target_dir)
                
            print(f"  └─ [완료] '{os.path.abspath(target_dir)}' 폴더에 저장되었습니다.")

        # ---------------------------------------------------------------------
        # [Case C] .gz 단일 파일 처리 구문 (단일 파일 바이너리 압축)
        # ---------------------------------------------------------------------
        elif file_name.endswith('.gz'):
            print(f"\n▶ [{file_name}] .gz 단일 압축 파일 해제 중...")
            
            # 슬라이싱 file_name[:-3]으로 끝의 '.gz'(3글자)만 잘라내어 해제될 파일명 생성
            # 예: image01.jpg.gz -> image01.jpg
            unpacked_filename = file_name[:-3]
            output_filepath = os.path.join(target_dir, unpacked_filename)
            
            # gzip.open(): 바이너리 읽기('rb') 모드로 압축 스트림을 읽음
            # open(): 바이너리 쓰기('wb') 모드로 압축 해제된 실제 원본 파일 생성
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_filepath, 'wb') as f_out:
                    # shutil.copyfileobj(): 입출력 스트림 간에 데이터를 블록 단위로 효율적으로 복사
                    shutil.copyfileobj(f_in, f_out)
                    
            print(f"  └─ [완료] '{os.path.abspath(output_filepath)}' 파일로 저장되었습니다.")

    print(f"\n[작업 완료] 지정된 모든 압축 파일의 해제가 성공적으로 처리되었습니다!")


# =============================================================================
# [실행부] 스크립트를 독립적으로 직접 실행할 때 작동하는 메인 블록
# =============================================================================
if __name__ == "__main__":
    # 1. 압축 파일이 위치할 입력 디렉토리와 압축을 풀 결과를 담을 출력 디렉토리 설정
    DATASET_DIR = "./dataset"        # 압축 파일(.zip, .tar.gz 등)을 넣어둘 경로
    EXTRACT_DIR = "./extracted_data"  # 해제된 원본 데이터(real/fake 폴더 등)가 풀려날 경로
    
    # 2. 압축 해제 메인 함수 호출
    extract_archives_in_dataset_dir(dataset_dir=DATASET_DIR, target_dir=EXTRACT_DIR)