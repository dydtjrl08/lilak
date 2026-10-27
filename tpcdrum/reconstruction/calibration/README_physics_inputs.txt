물리 보정 자료 수집과 Si 에너지 복원 (2026-10-27)

현재 확인된 범위
  기존 C12_72MeV_3C3_digi.root에는 관측 Si EdepSumSi branch가 없다.
  따라서 그 파일만으로 관측 Si 에너지에 기반한 3알파 질량 복원을 검증할 수 없다.
  C12_72MeV_3C2_digi_SiEdep_first10.root의 10개 이벤트에서는 8개 Si 모듈의
  EdepSumSiMeV 및 EdepSumSiDetectorNames를 실제로 읽었다. 이것은 3C3 검증이 아니다.
  실제 저장된 detector names는 SiDet_RU, SiDet_RM, SiDet_RD, SiDet_LU,
  SiDet_LM, SiDet_LD, SiDet_U, SiDet_D이다. SiDet_ 접두사를 임의로 제거하지 않는다.
  새 DDE digi는 아직 전달되지 않았다. 아래 경로는 새 파일을 만든 뒤 지정한다.

1. 서버에서 보정 파일과 설정만 모으기

  cd ~/workspace/lilak_seunghwan
  bash tpcdrum/reconstruction/calibration/collect_server_inputs.sh

  실행 결과의 Collected package 경로가 전송할 tar.gz이다. 표준 Python만 필요하며
  ripgrep, pandas, PyTorch는 필요 없다. 다수 GB의 digi는 이 압축에 넣지 않는다.
  기존 MC, digi, 설정, GEM gain을 수정하지 않는다. 출력 폴더는 항상 새 이름이어야 한다.

  새 digi가 준비되면 아래처럼 실제 절대경로를 추가한다.

  RECO_EXPORT_DIR=~/workspace/reco_inputs_20261027_dde \
  bash tpcdrum/reconstruction/calibration/collect_server_inputs.sh \
    --digi ~/workspace/lilak_seunghwan/tpcdrum/simulation/C12_DDE_allcomb_3C3_digi_SiEdep.root

  위 파일명은 예시이다. 실제 C12_digitize_hoyle_subset.C 출력 파일명으로 바꾼다.
  --digi는 event header, 저장된 parameter container만 읽는다. 전체 파형을 복사하거나
  MC의 알파별 에너지 정답을 추론 입력에 넣지 않는다.
  LILAK 없는 ROOT에서는 --skip-effective-parameters를 쓸 수 있으나, 해당 자료가 빠졌다고
  manifest에 표시되므로 완전한 보정 자료로 간주하면 안 된다.

2. 실제 Geant4 물질/geometry/dE/dx/range 추출

  평소 run_sim.exe가 실행되는 LILAK/NPTool/Geant4 환경에서 아래를 실행한다.
  라이브러리가 설치되지 않은 다른 Python 환경이나 이 Mac에서는 이 단계가 검증되지 않았다.

  cd ~/workspace/lilak_seunghwan
  python3 tpcdrum/reconstruction/calibration/export_build_run.py \
    --repo "$PWD" \
    --parameters tpcdrum/simulation/SimParameters.mac \
    --output ~/workspace/reco_materials_20261027

  build/CMakeFiles/.../run_sim.exe.dir/link.txt를 재사용해 새 calibration 실행 파일만 만든다.
  build 폴더 이름이 다르면 --build-dir을 추가한다. 후보가 여러 개면 오류에 나온 정확한
  --link-file 경로를 지정한다. --dry-run은 실행 없이 compiler/linker 명령만 저장한다.
  기존 run_sim.exe나 main object, simulation source는 수정하지 않는다.

  한 압축 안에 함께 모으려면 다음처럼 선택할 수 있다.

  RECO_EXPORT_DIR=~/workspace/reco_inputs_20261027_full \
  bash tpcdrum/reconstruction/calibration/collect_server_inputs.sh \
    --geant4-material-export \
    --digi ~/workspace/lilak_seunghwan/tpcdrum/simulation/C12_DDE_allcomb_3C3_digi_SiEdep.root

  안전을 위해 native exporter는 LKG4RunManager::Initialize의 MC 파일 RECREATE 부분을
  우회하고 G4RunManager의 geometry/physics 초기화만 호출한다. Run/BeamOn은 호출하지 않는다.
  해당 프로젝트의 실제 Geant4/NPTool 버전에 대한 컴파일/실행 확인은 서버에서 필요하다.

  주요 출력:
    geant4_materials.json : 실제 초기화된 밀도, 압력, 온도, X0, 물질/region identity,
                           Si 활성면 위치/회전/크기/두께, Geant4 alpha/C12 질량,
                           alpha energy/dE/dx/range arrays
    material_N_alpha.csv : MeV, MeV/mm, mm
    export_receipt.json  : parameter/source/link 명령 및 해시, 생성 이벤트 수 0
    compile.log/link.log/export.log : 실패 시 확인

  파일명의 650Torr만으로 실제 transport 밀도를 단정하지 않는다. 전달된 과거 source에서는
  GasPressure=650을 읽지만 혼합 gas가 STP pressure/density로 생성되는 부분이 확인되었다.
  현재 서버에서 수정됐을 수 있으므로 exporter가 초기화한 실제 G4Material 값을 확인한다.
  이 도구는 압력, 가스 조성, GEM gain을 자동 수정하지 않는다.

  ComputeTotalDEDX와 GetCSDARange는 독립적인 Geant4 정의이다. 전자는 적용 가능한
  nuclear stopping을 포함한다. 사용 에너지 범위에서 StoppingTable.consistency_audit()로
  range 기울기와 dE/dx를 비교하고, 실제 transport의 mean-loss closure를 검증한다.
  배열이 출력됐다는 사실만으로 event별 에너지 복원이 정확하다고 선언하지 않는다.

3. 각 이벤트의 재구성 궤적에 따른 물질 경로

  정적 gas 길이 하나를 모든 이벤트에 재사용하지 않는다. 추론으로 얻은 vertex, 관측
  TPC 궤적, Si 입사점으로 path를 구성한다. MCStep 정답 궤적은 추론 경로로 사용하지 않는다.
  각 event/track에 고유한 정수 path_id를 부여하고 vertex -> Si 순서로 segment_id를 부여한다.
  CSV header는 정확히 아래와 같아야 한다. 좌표는 pad (u,v,w) mm이다.

    path_id,segment_id,u0,v0,w0,u1,v1,w1

  마지막 점은 Si 활성층 입구이다. Si 중앙면에 멈추면 Si 반두께를 gas로 잘못 세게 된다.
  dead layer/window가 실제 geometry에 없으면 별도 측정된 물질과 두께를 명시한다.

  python3 tpcdrum/reconstruction/calibration/export_build_run.py \
    --repo ~/workspace/lilak_seunghwan \
    --parameters tpcdrum/simulation/SimParameters.mac \
    --path-segments /absolute/path/reconstructed_path_segments.csv \
    --output ~/workspace/reco_material_paths_20261027

  G4Navigator가 segment를 실제 물질 경계별로 나누며 reco_material_paths.csv에 다음을 쓴다.
    path_id,segment_id,part,material,length_mm,volume
  zero step, world 이탈, 누락된 물질표는 성공으로 처리하지 않는다.
  Python load_material_paths(csv, expected_segment_lengths_mm)로 제출한 모든 segment와
  길이 합을 검증한 후 inverse_energy(..., path_validated=True)에 전달한다.
  export_receipt.json의 path 입력/출력 해시를 보존하고 원래 event/track 매핑도 함께 저장한다.

4. Si 관측 에너지와 TPC 궤적 연결의 조건

  입력: EdepSumSiDetectorNames, EdepSumSiMeV, 관측으로 재구성한 궤적/vertex/공분산.
  EdepSumSiTrackID/TrackMeV/TrackDetID 같은 알파별 MC branch는 평가 정답 전용이다.
  서로 다른 모듈에 한 알파씩 입사하면 geometry association으로 모듈 Edep를 연결한다.
  여러 알파가 같은 모듈에 들어오면 모듈 총합 하나로 개별 에너지들을 유일하게 구할 수 없다.
  코드는 동일 분배를 강제하지 않고 unresolved_shared_detector 상태를 반환한다.
  독립적으로 보정된 관측 TPC 전하 에너지 prior가 있을 때만 shared_si_energy_fit으로
  분배 가설을 결합할 수 있다. MC 에너지를 prior로 쓰면 정답 누수이다.

  Edep=T_at_Si는 완전히 정지하고 에너지가 전부 수집된 경우의 조건부 관계이다.
  stopping_verified 또는 검증된 kinetic upper bound와 range가 없으면 punch-through와
  구분하지 못하므로 silicon_entrance_energy는 미해결 상태를 반환한다.
  Si dead layer, escape energy, 에너지 calibration과 그 covariance가 필요하다.

5. TPC z/time, energy uncertainty를 위해 추가로 필요한 것

  - 실제 사용한 SimParameters/저장된 digi parameter container 및 Garfield/field/pulse/noise assets
  - charge-mean time 정의에 맞춘 drift velocity, t0/trigger/pulse response timing calibration
    단순 TB*0.36은 물리 w가 아니다. 위 raw asset 추출만으로 timing calibration이 끝나지 않는다.
  - 관측 신호의 pedestal/noise와 pad response, 측정 공분산
  - Si module별 활성면/rotation/dead layer, 에너지 response/covariance와 stopping 판정
  - 물질별 mean-loss/straggling 자료와 실제 경로 및 밀도/두께 불확실도
  - 에너지가 낮은 구간이나 table 범위 밖이면 더 넓은 유효 물리표를 먼저 준비

  inverse_energy는 mean CSDA inversion이며 알 수 없는 event별 straggling을 역으로 제거하지
  않는다. boundary energy 및 선택적으로 제공한 correlated path-length covariance만 전파한다.
  smoother의 energy prior는 first observed point에서의 kinetic energy이다. Si 에너지나
  vertex 에너지를 그 자리에 직접 넣지 말고 해당 경로까지 먼저 mean-loss 역전파한다.

6. Mac으로 전송 (Mac 터미널에서)

  mkdir -p "/Users/yongseok/Documents/Codex/2026-09-02/uniform-digitization-charge-centroid/received_data/kalman_inputs"
  scp -P 5522 \
    yongseok@210.117.209.247:/home/yongseok/workspace/reco_inputs_20261027_full.tar.gz \
    "/Users/yongseok/Documents/Codex/2026-09-02/uniform-digitization-charge-centroid/received_data/kalman_inputs/"

  실제 출력 archive 이름으로 바꾼다. collection_manifest.json의 missing/skipped/errors도
  함께 전달한다. 새 3C3 DDE digi는 별도 전송하며 현재 존재한다고 가정하지 않는다.

7. 질량/상관각 검증

  alpha kinetic T와 단위 방향 n으로 p=sqrt(T(T+2m_alpha))*n, E=T+m_alpha를 쓴다.
  P=sum p_alpha, M=sqrt(P_E^2-|P_vec|^2), E_x=M-m_C12_ground 이다.
  7.654 MeV는 C12 전체 질량이 아니라 바닥 상태에 대한 여기 에너지이다.
  각 alpha를 beta=P_vec/P_E의 음수로 boost하여 COM 에너지/상관각/Dalitz를 계산한다.
  독립적인 결과에는 120도나 에너지 등분배를 강제하지 않는다.
  DDE truth의 120도는 소프트웨어 closure fixture이며 실제 reconstruction 성공 판정과 분리한다.

검증 기록
  Python analytic/synthetic physics tests 13개 통과.
  stdlib collector/build preparation tests 2개 통과.
  ROOT 6.40.02에서 graph/pulse/header round-trip fixture 및 실제 10-entry SiEdep header 확인.
  ROOT TLorentzVector의 독립 계산과 NumPy four-vector/COM boost/mass/상관각 결과 일치 확인.
  Geant4/NPTool material exporter의 서버 컴파일, 실제 물질 경로, 물리 보정 closure는 아직 미검증.
