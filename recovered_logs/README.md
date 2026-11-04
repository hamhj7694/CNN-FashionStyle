# V1 학습 로그 복구

- v1_training_log.txt: VS Code 로컬 기록에서 복구한 실제 1~9 epoch 출력.
- v1_training_metrics.csv: 해당 출력에서 추출한 손실·정확도. 정확도 단위는 %이며 원래 출력의 반올림된 값이다.
- v1_training_cell.py: 2026-11-04 19:49:06 로컬 기록에서 복구한 기존 학습 셀 코드. 모듈·모델·데이터 준비 셀 실행 후 사용하는 코드이며 단독 스크립트가 아니다.
- training_recovery.html: 로그와 기존 V1 노트북의 정확도 그래프를 함께 볼 수 있는 파일.

10번째 epoch의 학습 로그는 확인한 기록에서 찾지 못했다. 저장 모델의 최종 가중치만으로 과거 epoch별 손실·정확도를 복원할 수는 없다.

현재 V1 노트북에는 학습 코드 대신 출력 문구가 들어간 셀이 있다.
`model_device=cuda:0, ...`는 Python 실행 코드가 아니므로 문법 오류가 발생한다.
실제 학습 반복문이 필요하다면 복구된 v1_training_cell.py의 내용을 확인한다.
과거 로그를 보기 위해 모델 생성/학습 셀을 다시 실행할 필요는 없다.

V2 수정 시 복사된 이전 실행 출력은 초기화했지만 V1 노트북은 수정하지 않았다.
이번 복구 작업에서도 원본 노트북과 가중치 파일을 덮어쓰지 않았다.

프로젝트 PowerShell에서 열기:
```powershell
Invoke-Item ".\recovered_logs\training_recovery.html"
```
