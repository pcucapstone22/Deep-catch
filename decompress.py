import os
import tarfile
import gzip
import shutil

def extract_archives_in_dataset_dir(dataset_dir="./dataset", target_dir="./extracted_data"):
    """
    지정된 'dataset' 디렉토리 내를 스캔하여 
    존재하는 모든 .tar.gz 및 .gz 형태의 압축 파일을 해제하는 함수
    
    :param dataset_dir: 압축 파일들이 들어있는 폴더 경로 (기본값: ./dataset)
    :param target_dir: 압축 해제된 결과물이 저장될 폴더 경로 (기본값: ./extracted_data)
    """
    # [1단계] 'dataset' 디렉토리 존재 여부 확인 및 자동 생성
    # 만약 지정한 dataset 폴더가 없다면 폴더를 새로 만들고 안내문 출력 후 종료
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"[{dataset_dir}] 디렉토리가 새로 생성되었습니다.")
        print(f"-> 압축을 풀 .tar.gz 또는 .gz 파일을 '{os.path.abspath(dataset_dir)}' 폴더에 넣고 다시 실행하세요.")
        return

    # [2단계] 'dataset' 폴더 내의 파일 목록 탐색 및 압축 파일 필터링
    # os.listdir()로 디렉토리 내부의 모든 파일 및 폴더 이름 가져오기
    all_files = os.listdir(dataset_dir)
    
    # 확장자가 .tar.gz, .tgz, .gz 로 끝나는 압축 파일만 리스트로 수집
    archive_files = [f for f in all_files if f.endswith(('.tar.gz', '.tgz', '.gz'))]

    # 탐색된 압축 파일이 하나도 없을 경우 예외 처리
    if not archive_files:
        print(f"[{dataset_dir}] 폴더 내에 해제할 압축 파일(.tar.gz 또는 .gz)이 없습니다.")
        print(f"현재 폴더 내용물: {all_files}")
        return

    print(f"[{dataset_dir}] 디렉토리에서 총 {len(archive_files)}개의 압축 파일을 발견했습니다.")
    
    # [3단계] 압축 해제된 파일들을 담을 대상 디렉토리(target_dir) 생성 
    os.makedirs(target_dir, exist_ok=True)

    # [4단계] 발견된 압축 파일들을 하나씩 순회하며 압축 해제 수행
    for file_name in archive_files:
        # 파일의 전체 상대 경로 결합 (예: ./dataset/my_images.tar.gz)
        file_path = os.path.join(dataset_dir, file_name)
        
        # --- Case A: .tar.gz 또는 .tgz 파일 (묶음 파일 아카이브) ---
        if file_name.endswith(('.tar.gz', '.tgz')):
            print(f"\n▶ [{file_name}] .tar.gz 아카이브 해제 시작...")
            
            # tarfile.open(): gzip 읽기 전용 모드('r:gz')로 파일 열기
            with tarfile.open(file_path, "r:gz") as tar:
                # tar.extractall(): 지정한 target_dir 경로에 내부 모든 구조 해제
                tar.extractall(path=target_dir)
                
            print(f"  └─ 완료: '{os.path.abspath(target_dir)}' 폴더에 저장됨")

        # --- Case B: .gz 단일 압축 파일 ---
        elif file_name.endswith('.gz'):
            print(f"\n▶ [{file_name}] .gz 단일 파일 해제 시작...")
            
            # 파일 이름 끝의 '.gz'(3글자)를 잘라내어 해제 후 파일명 생성 (예: image.jpg.gz -> image.jpg)
            unpacked_filename = file_name[:-3]
            output_filepath = os.path.join(target_dir, unpacked_filename)
            
            # gzip.open()으로 바이너리 읽기('rb') 후 파일 작성 모드('wb')로 데이터 복사
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            print(f"  └─ 완료: '{os.path.abspath(output_filepath)}' 파일로 저장됨")

    print(f"\n[작업 완료] 모든 압축 파일의 해제가 성공적으로 완료되었습니다!")


# ========================================================
# [실행부] 스크립트를 직접 실행할 때 작동하는 코드
# ========================================================
if __name__ == "__main__":
    # 1. 압축 파일이 담긴 입력 디렉토리와 해제 결과를 담을 출력 디렉토리 설정
    DATASET_DIR = "./dataset"        # 압축 파일들이 담길 폴더
    EXTRACT_DIR = "./extracted_data"  # 해제된 이미지가 저장될 폴더
    
    # 2. 압축 해제 메인 함수 실행
    extract_archives_in_dataset_dir(dataset_dir=DATASET_DIR, target_dir=EXTRACT_DIR)
