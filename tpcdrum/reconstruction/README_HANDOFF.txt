관측 전용 ROOT 연결과 고정 표본 평가

기본 실행 방법은 README.txt와 reconstruction_control.ipynb를 따른다.
simulation/C12_run_reconstruction.C에서 finding, fitting, vertex 등 각 단계를
따로 실행하거나 전체 단계를 실행할 수 있다. 새 출력 경로를 사용한다.

같은 finding 예측을 평가와 후속 재구성에 함께 사용하기
  training_finding.evaluate_finding_comparison(...,
      observation_outputs={method: new_root_path, ...})
  observation_outputs의 method 집합은 config.methods와 같아야 한다.
  각 파일은 원래 예측 슬롯과 positive waveform, component/background,
  관측된 Si 모듈 이름·에너지를 저장한다. MC 정답, 평가용 순열 정렬과 점수는
  이 관측 파일에 넣지 않는다. 별도 evaluation ROOT에만 저장한다.
  따라서 예측을 다시 실행하지 않고 같은 관측 ROOT를 fitting과 vertex에 넣는다.

실패와 적은 트랙 수
  K=0으로 정상적으로 응답한 방법은 실제 누락 전하에 대한 오차를 받는다.
  실행 실패로 예측이 없는 이벤트는 K=0 placeholder와 명시적 실패 플래그로 남긴다.
  이 placeholder의 배경=q는 물리적 배경 예측을 뜻하지 않는다.
  실패의 전하 오차·각도·에너지·vertex 정확도는 NaN이며, 원래 이벤트 분모와
  실패 수는 유지한다. 성공한 일부 이벤트의 조건부 정확도와 coverage를 함께 본다.
  찾은 트랙이 없는 경우 fitting 성공률은 0이며 vertex를 만들지 않는다.

고정된 시험 표본
  train/validation/test는 원래 source-event 단위로 나눈다.
  angular_benchmark.benchmark의 config.event_keys에 시험 이벤트의
  [file_index, entry] 목록을 명시하면 그 표본 전체를 요구한다.
  이를 생략하면 truth ROOT의 모든 이벤트가 비교 대상이다.
  같은 이벤트의 세 알파 또는 세 쌍을 독립 표본으로 취급하지 않는다.

서버 환경
  CUDA 서버의 finding.device는 "cuda" 또는 명시한 "cpu"다.
  macOS용 "mps" 설정을 서버로 그대로 복사하지 않는다.
  PyTorch 1.10에서 자신이 학습하고 해시를 확인한 체크포인트를 읽을 때는
  finding.trusted_checkpoint=true를 명시한다. 최신 PyTorch는 제한 로더를 쓴다.
  모델 파일은 이 소스 묶음에 포함하지 않는다.
  NumPy/SciPy 등의 환경은 requirements.txt에 맞춰 서버에서 설치한다.

검증 범위
  소프트웨어 검사는 생성한 ROOT fixture의 슬롯·실패·독립 재읽기를 검증한다.
  서버 ROOT/PyTorch 버전의 실제 실행과 물리적 정확도는 별도로 확인해야 한다.
  waveform 시간의 물리적 z 변환과 Si 역산에는 검증된 보정·재료 경로가 필요하다.
  120도 또는 에너지 등분배를 재구성 결과에 강제하지 않는다.
