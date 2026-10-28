TPCDrum 단계별 재구성 개발판 — 2026-10-27

현재 상태
  Sequential clean-seed/soft-peel/두-track vertex/세 번째 track/공동 피팅 추가 사항은
  README_SEQUENTIAL.txt를 참고한다. 기본 config는 이 새 경로를 사용한다.
  실제 기존 3C3 digi를 읽고 모든 단계 ROOT 저장/재시작을 확인했다.
  새 equal-energy DDE digi는 아직 만들어지지 않았다. 기존 3C3에는 관측 Si 에너지
  branch가 없어 그 파일로 새 DDE의 불변질량/120도 복원이 검증됐다고 할 수 없다.
  구현·수학 검증과 실제 검출기 성능을 REQUIREMENTS.json에서 따로 기록한다.
  기본 config에는 보정값을 임의로 채우지 않았다. missing-input 상태도 ROOT에 남긴다.

입력 파일 구분
  C12_make_hoyle_subsets.C가 만든 3C3.root: digitization 전 MC subset.
  C12_digitize_hoyle_subset.C: 변환을 수행하는 매크로(.C).
  그 매크로의 *_digi_SiEdep.root: 이 reconstruction이 읽을 관측 입력.
  사용자 예정 위치: ~/workspace/lilak_seunghwan/tpcdrum/simulation.
  config의 C12_DDE_allcomb_3C3_digi_SiEdep.root는 예시 이름이다.
  새 digi에는 RawPad, EdepSumSiMeV, EdepSumSiDetectorNames가 필요하다.
  알파별 MC energy/position/tag는 별도 평가 정답에만 사용한다.

설치
  다운로드한 별도 clone에서 다음을 실행한다. 기존 서버 파일이 있으면 멈춘다.
  python3 tpcdrum/reconstruction/install_new_reconstruction.py \
    ~/workspace/lilak_seunghwan
  복사 대상은 새 reconstruction 폴더와 simulation의 새 C12_* 매크로 2개뿐이다.
  git pull/reset/clean, 기존 task/config/data의 수정은 하지 않는다.

환경
  ROOT와 현재 CNN Python을 함께 쓸 수 있는 동일한 환경/컨테이너에서 실행한다.
  python3 -m pip install -r ../reconstruction/requirements.txt
  PyTorch는 기존 GPU 환경을 사용한다. geometry/physics 단계만 쓰면 torch는 선택이다.
  Jupyter의 sys.executable 출력이 사용할 Python 경로다. 컨테이너의 Python 경로를
  호스트 ROOT에 넣어도 실행되지 않는다. 필요하면 함께 제공한 노트북을 쓴다.
  CLI 자체는 run.py에 package 경로를 설정하므로 PYTHONPATH 설정이 필요 없다.

설정과 첫 실행
  config.json을 새 run_config.json으로 복사한 뒤 input_roots를 실제 digi로 바꾼다.
  경로는 config 파일이 있는 폴더 기준이다. entries=[0,5]는 작은 점검 표본이다.
  entries=null은 모든 이벤트이며 단계마다 full waveform이 저장되므로 저장공간을 확인한다.
  동일 출력 경로는 덮어쓰지 않는다. 매 실행마다 새 results 폴더 이름을 쓴다.

  cd ~/workspace/lilak_seunghwan/tpcdrum/simulation
  root -l -b -q 'C12_run_reconstruction.C("inspect","","","../reconstruction/run_config.json","/root/miniconda3/bin/python")'
  root -l -b -q 'C12_run_reconstruction.C("all","","../reconstruction/results/dde_first_run","../reconstruction/run_config.json","/root/miniconda3/bin/python")'

  위 Python 경로는 이전 Jupyter 환경의 예시다. 실제 sys.executable로 바꾼다.
  보정/Si 정보가 부족하면 all은 ROOT 진단 파일들을 남기고 종료코드 4를 반환한다.
  이때 ROOT launcher는 실패로 표시한다. 계산된 척한 각도/에너지를 만들지 않는다.

단계별 실행
  단계명: ingest, prepare, spacepoints, finding, fitting, vertex, association,
          material_request, material_import, energy, refitting, kinematics.
  all은 순서대로 실행한다. refitting.enabled=true일 때 energy 뒤에 재피팅하고
  vertex/Si 연결/물질 경로/에너지를 다시 계산한다. 변경된 궤적에 옛 경로를 재사용하지 않는다.
  spacepoints는 representation="space_points"일 때만 all에 포함된다.

  python /absolute/path/tpcdrum/reconstruction/run.py --stage finding \
    --config run_config.json --input 01_prepare.root --output finding_new.root
  root -l 'C12_view_reconstruction.C("finding_new.root",0)'
  root -l -b -q 'C12_view_reconstruction.C("finding_new.root",0,"entry0")'
  마지막 호출은 이벤트/12개 레이어 파형/진단 PDF를 저장한다. 다음 이벤트는 row를 바꾼다.

ROOT 저장 계약
  events: 원래 관측 파형, 현재와 이전 단계의 모든 숫자 배열, event identity, record_json.
  manifest: schema/config/calibration provenance/완료 여부.
  metrics 및 performance/: 단계 유효성·누락·모호성·진단 히스토그램.
  숫자 배열은 primitive jagged branch와 <name>__shape branch에 저장한다.
  예: components=(K,12,64,512), fit_states=(K,12,5), vertex_uvw_mm=(3,).
  다음 단계는 이 ROOT만 읽는다. 이전 ROOT, 원본 digi, 이미 수행한 모델은 필요 없다.
  현재 단계가 새로 쓸 checkpoint/보정값은 명시적으로 지정한다.
  ROOT6.24 호환을 위해 TTree를 쓴다. Mac ROOT6.40에서 native 읽기/보기는 검증했으나
  서버의 실제 ROOT6.24·Geant4·PyTorch1.10 조합은 서버 확인이 필요하다.

두 표현 경로
  direct_waveform: 전체 native pad×layer×512TB -> CNN/geometry/waveform 가설.
  space_points: pad별 pulse island 중심점 -> Hough/RANSAC -> point 분배.
  space-point 변환은 동일 pulse island 안의 중첩 정보를 잃을 수 있다.
  두 경로 모두 동일한 관측 identity로 성능을 비교한다.

Finding 설정
  finding.method = hough / ransac / tls / cnn / cnn_hough / waveform_hypotheses.
  CNN은 native-pad ResUNet3D + gated skip + layer/time attention, PIT 학습이다.
  새 구조는 훈련된 checkpoint를 명시해야 한다. 임의 가중치를 대신 쓰지 않는다.
  waveform_hypotheses는 이웃 패드와 앞뒤 레이어를 함께 쓰는 forward model을 적합한다.
  whole-pad holdout에서 예측 신호를 검증하고, 가설별 잔차·조건수·미식별 상태를 저장한다.
  measured charge 분배가 합을 보존한다는 사실만으로 올바른 분리를 증명할 수 없다.
  서로 같은 응답의 완전 중첩은 개별 전하를 유일하게 구하지 못하며 ambiguous로 남는다.
  compare_k=true는 명시된 K 가설의 비교이며 자동 multiplicity 정답을 보장하지 않는다.

Fitting 설정
  linear_rts: 4D (u,w,du/dv,dw/dv), 에너지를 모르는 기준 모델.
  ekf_rts / ukf_rts: 5D 에너지와 물질 손실을 포함한 확장/무향 Kalman + RTS.
  DAF/Huber 방식의 반복 가중치로 이상치를 줄일 수 있으나 이것만으로 분리를 검증하지 않는다.
  legacy_sequence_cnn: 사용자가 훈련한 sequence ResNet을 명시적으로 가져오는 선택 모드.
  projected initial head와 첫 관측 local chord를 구분한다. CNN의 covariance는 임의로 만들지 않는다.
  물리적 w에는 charge-mean timing에 맞는 drift/t0/pad delay 보정이 필요하다.
  구름 폭/샘플링 covariance는 측정 분해능의 대용 가정이다. ADC로 나눠 과도하게 줄이지 않는다.

Si, 물질 경로, 에너지
  calibration/README_physics_inputs.txt의 서버 수집·Geant4 export 명령을 사용한다.
  navigator.enabled=true면 명시된 서버 repo/parameters로 native geometry와 물질 경로를
  호출한다. 기존 simulation 실행 파일/가스/GEM gain을 바꾸거나 이벤트를 생성하지 않는다.
  exporter의 실제 서버 컴파일은 아직 검증되지 않았다. 먼저 작은 표본에서 확인한다.
  Si 연결은 마지막 local tangent와 실제 활성층 앞면을 사용한다. 검출기 이름으로 정렬한다.
  같은 Si 모듈에 여러 알파가 찍히면 관측 총합 하나로 에너지를 등분배하지 않는다.
  경로는 vertex -> 관측점들 -> Si 입구이며 G4Navigator가 실제 물질로 나눈다.
  request CSV, native export receipt, 물질표의 해시와 길이 합을 검증한 후 ROOT에 포함한다.
  이후 energy 단계는 외부 CSV/JSON을 지워도 저장된 ROOT만으로 실행할 수 있다.
  mean CSDA 역산은 사건별 straggling을 되돌리지 못한다. Si stopping 및 보정값도 필요하다.
  first-observed-point의 kinetic energy와 vertex energy는 별도 branch다.

운동학과 평가
  M=sqrt((sum E)^2-|sum p|^2), Ex=M-m(C12 ground).
  7.654MeV는 M 자체가 아니라 Ex다. 합 4-vector로 COM boost한 뒤 pair angles/Dalitz를 계산한다.
  120도나 에너지 등분배를 재구성 결과에 강제하지 않는다.
  첫 관측점 방향을 vertex 방향과 같다고 할 경우 config에서 그 근사를 명시해야 한다.
  기본값은 candidate만 계산하며 initial direction 복원 성공으로 세지 않는다.
  분리 결과가 ambiguous이거나 그 판정이 누락/손상되면, 유한한 질량·각도 값은
  candidate로만 보존한다. kinematics_ok와 kinematics_method_accepted는 모두
  명시적으로 유효한 비모호 finding 및 에너지·방향 조건이 있어야 1이 된다.
  이 판정 자체가 물리적 정확도나 확률 보정 완료를 뜻하지는 않는다.
  전파할 9x9 full joint covariance를 명시한 경우에만 운동학 오차를 계산한다.

  평가 정답은 별도의 truth.root다. inference 입력으로 사용할 수 없다.
  python run.py --stage truth --config run_config.json --output truth.root
  python run.py --stage evaluate --config run_config.json \
    --input finding_new.root --truth-root truth.root --output evaluation.root
  root -l 'C12_view_reconstruction.C("evaluation.root")'
  이벤트 전체에 하나의 alpha 순열을 매칭하고 배경은 고정한다. 레이어마다 정답 순서를
  바꾸어 성능을 높이지 않는다. 누락된 방향은 NaN과 coverage로 기록한다.
  평가 ROOT는 다음 추론 단계 입력으로 쓸 수 없다.

검증 범위와 남은 일
  기존 3C3 48개 source-group pilot: train34/validation7/test7 고정, 작은 CNN 5epoch.
  validation-best checkpoint를 고른 뒤 시험했다. 작은 pilot이므로 최적 성능 결론은 아니다.
  새 DDE digi, 실제 timing·PRF·Si 보정과 native geometry export가 오면 같은 분할에서
  궤적·vertex·질량·COM 상관각 및 불확실도/coverage를 끝까지 다시 검증해야 한다.
  자세한 알고리즘 근거는 references/, 구현 검사는 checks/에 있다.

저장된 시간 설정 확인
  calibration/read_stored_parameters.py는 실제 digi의 top-level parameter
  container만 읽고 원문/평가값/중복/metadata key 해시를 저장한다. 파형이나 MC event는
  읽지 않는다. 실제 저장 설정의 감사 결과는 로컬 자료로 별도 보관한다.
  이 수치만으로 charge-mean 시간의 절대 z 보정이 검증되었다고 처리하지 않는다.
  pulse template 평균시간, field-dependent drift와 당시 asset 파일 내용이 더 필요하다.
  현재 설정과 당시 저장된 참조를 구분해 모으는 명령은
  calibration/README_timing_audit.txt에 있다. 가스/GEM gain은 변경하지 않는다.

관측 ROOT handoff와 고정 표본 평가의 추가 설명
  README_HANDOFF.txt: 같은 예측의 단계별 저장, K=0/실패와 coverage, 서버 환경.
  calibration/README_empirical_timing.txt: inference와 분리된 MC timing 연구 도구.
