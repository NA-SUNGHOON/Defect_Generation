## Defect Generation Tool (AI 학습용 결함 합성 툴)

### 설치

1. Python 3.10 이상 설치
2. 이 디렉터리에서 다음 명령 실행:

```bash
pip install -r requirements.txt
```

### 실행

```bash
python app.py
```

### 주요 기능 (첫 버전)

- 이미지 불러오기
- 스크래치 / 점 결함 자동 생성 (랜덤 파라미터)
- 생성된 결함 리스트(검사 리스트) 확인
- 결함이 합성된 이미지를 저장
- 결함 메타데이터(JSON) 저장 (타입, 위치, 크기 등)

추후 COCO / YOLO 라벨 포맷 지원, 더 다양한 결함 타입, 파라미터 UI 등을 확장할 수 있습니다.


