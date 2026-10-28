물리 보정 자료 수집과 Si 에너지 복원

이 문서는 공용 소스 사용 안내다. 개인 서버 주소, 실제 입력 파일의 metadata,
보정 계수, 이벤트 목록이나 새 검출기 성능 결과를 포함하지 않는다.
코드는 새 delivery clone에서 실행하고, 기존 simulation checkout은 읽기 대상으로
지정한다. 기존 checkout에 pull하거나 파일을 덮어 설치할 필요가 없다.

아래 환경 변수는 사용자가 자신의 실제 경로로 지정한다.
  RECO_REPO      : 초기화된 기존 LILAK/NPTool simulation checkout
  RECO_DIGI      : 실제 존재하는 digitized ROOT 파일
  RECO_OUTPUT    : 아직 존재하지 않는 새 출력 디렉터리
  RECO_REMOTE    : 자신의 SSH user@host
  RECO_ARCHIVE   : collector가 출력한 서버상의 압축 파일 절대경로
  RECO_LOCAL_DIR : 수신할 새 로컬 디렉터리
  RECO_SSH_PORT  : 자신의 SSH 포트

1. 설정과 자산을 구분해서 모으기

새 소스의 tpcdrum/reconstruction 디렉터리에서 실행한다.

  python3 calibration/export_collect_inputs.py \
    --repo "$RECO_REPO" \
    --parameters "$RECO_REPO/tpcdrum/simulation/SimParameters.mac" \
    --digi "$RECO_DIGI" \
    --asset-root "$RECO_REPO/tpcdrum/common" \
    --output "$RECO_OUTPUT"

collector 자체는 표준 Python을 사용한다. native ROOT metadata export에는 해당
환경의 ROOT/LILAK가 필요하다. 전체 digi나 파형은 압축에 복사하지 않는다.
현재 설정 파일이 참조한 자산과 저장된 ROOT parameter가 참조한 자산을 따로
기록한다. --asset-root는 선택 사항이며, 같은 이름의 후보 파일을 찾을 때만 쓴다.
후보가 존재하거나 해시를 계산했다는 사실은 과거에 쓰인 자산과의 동일성 증명이 아니다.

기존 MC/digi, 설정, 가스 조성, 압력이나 GEM gain을 수정하지 않는다.
출력 디렉터리 또는 동명의 archive가 있으면 덮어쓰지 않고 멈춘다.
누락·권한 오류·크기 제한은 collection manifest와 로그에 남긴다.
--skip-effective-parameters로 생략한 설정은 검증 완료로 간주하지 않는다.
실제 digi가 준비되지 않았다면 존재하지 않는 예상 파일명을 입력하지 않는다.

2. 초기화된 Geant4 물질과 geometry 추출

원래 simulation이 동작하는 LILAK/NPTool/Geant4 환경에서 실행한다.
이 exporter의 대상 서버 버전별 컴파일·실행은 별도로 검증해야 한다.
파일명이나 설정값만으로 실제 transport 밀도·온도·압력을 확정하지 않는다.

  python3 calibration/export_build_run.py \
    --repo "$RECO_REPO" \
    --parameters "$RECO_REPO/tpcdrum/simulation/SimParameters.mac" \
    --output "$RECO_OUTPUT"

각 작업마다 RECO_OUTPUT을 별개의 새 경로로 지정한다. --build-dir은 기존 build
디렉터리 이름이 다를 때 사용한다. link 후보가 여러 개면 --link-file로 정확한
기존 link.txt를 지정한다. --dry-run은 compiler/linker 명령만 준비한다.
원래 실행 파일, main object와 simulation source는 바꾸지 않는다.
native 도구는 geometry/physics만 초기화하며 Run/BeamOn으로 이벤트를 생성하지 않는다.
MC 출력 파일을 재생성하는 초기화 경로도 우회한다.

collector에 --geant4-material-export를 추가하면 같은 새 collection 안에 이
추출 작업을 요청할 수 있다. 실패 로그도 보존한다. export 파일이 만들어졌다는
사실만으로 실제 물리 보정이나 사건별 에너지 복원이 검증된 것은 아니다.

출력은 초기화된 G4Material, 활성 Si geometry, 질량, alpha stopping/range 표와
source/build/export receipt를 포함한다. 이 출력은 로컬 연구 자료이며 공용
소스 저장소에 함께 게시하지 않는다. 표의 유효 범위와 단위, range-dE/dx 일관성,
실제 transport의 mean-loss closure를 별도로 확인한다.

3. 관측으로 복원한 궤적의 물질 경로

정적인 gas 길이 하나를 모든 사건에 재사용하지 않는다. 추론으로 얻은 vertex,
TPC 궤적과 Si 입사점으로 경로를 만든다. MCStep 정답은 추론 경로로 넣지 않는다.
event/track별 고유 정수 path_id와 vertex -> Si 순서의 segment_id를 사용한다.
path-segment CSV header는 다음과 같으며 좌표는 pad (u,v,w) mm다.

  path_id,segment_id,u0,v0,w0,u1,v1,w1

마지막 점은 Si 활성층 입구다. 중앙면까지의 경로를 gas로 세지 않는다.
dead layer/window가 geometry에 없다면 측정한 물질과 두께를 별도로 명시한다.
위 exporter의 --path-segments 인자에 실제 CSV를 주면 navigator가 물질 경계를
따라 경로를 나눈다. world 이탈, zero step, 누락된 물질표는 성공이 아니다.
제출한 모든 segment와 길이 합을 검증한 뒤 inverse_energy에 전달한다.
입력/출력 receipt와 원래 event/track 매핑을 함께 보존한다.

4. Si 에너지 연결과 독립적인 역산 조건

추론에는 관측 모듈 이름과 모듈별 총 Edep, 관측으로 재구성한 궤적·vertex 및
그 공분산을 사용한다. 모듈 이름과 에너지 배열의 순서를 그대로 유지한다.
알파별 MC track ID나 MC 에너지는 평가 정답 전용이다.

서로 다른 모듈에 한 알파씩 입사하면 geometry association으로 모듈 Edep를
연결할 수 있다. 여러 알파가 같은 모듈에 들어오면 총합 하나로 개별 에너지들을
유일하게 정할 수 없다. 동일 분배를 강제하지 않고 unresolved로 남긴다.
독립적으로 보정한 관측 TPC 에너지 prior가 있을 때만 분배 가설을 결합한다.
MC 에너지를 prior로 사용하는 것은 정답 누수다.

Edep = Si 입구 운동에너지는 입자가 완전히 정지하고 에너지가 모두 수집된
경우의 조건부 관계다. stopping/punch-through 판정, dead layer, escape energy,
에너지 보정과 공분산이 필요하다. 검증된 Si 입구 에너지와 실제 물질 경로를
역으로 적분하여 vertex 에너지를 구한다. mean CSDA inversion은 알 수 없는
사건별 straggling을 역으로 제거하지 않는다.
smoother의 energy prior는 first observed point의 운동에너지다. Si 또는 vertex
에너지를 그 자리에 그대로 넣지 않고 해당 지점까지의 물질 경로를 먼저 처리한다.

5. 시간·위치와 불확실도

관측 charge-mean과 같은 시간 통계량에 맞춰 drift response, t0/trigger/pulse
timing을 보정해야 한다. raw 자산을 모았다는 사실만으로 시간-물리 좌표 변환이
완료되지는 않는다. pedestal/noise, pad response, 측정 공분산과 물질 경로·밀도·
두께 및 에너지 손실의 불확실도도 필요하다. 표의 범위를 벗어나면 계산을 강행하지 않는다.

6. 자기 서버에서 새 로컬 폴더로 전송

collector가 출력한 실제 archive 경로를 RECO_ARCHIVE에 지정한다.
로컬 터미널에서 새 수신 폴더만 만든 뒤 다음을 실행한다.

  if mkdir "$RECO_LOCAL_DIR"; then
    scp -P "$RECO_SSH_PORT" "$RECO_REMOTE:$RECO_ARCHIVE" "$RECO_LOCAL_DIR/"
  fi

이미 존재하는 폴더에 덮어 받지 않는다. collection의 missing/skipped/errors도
함께 전달한다. 대용량 digi 전송은 이 작은 보정 자료 collection과 별도로 정한다.

7. 질량과 상관각 검증

각 alpha의 운동에너지 T와 단위 방향 n으로 p=sqrt(T*(T+2*m_alpha))*n,
E=T+m_alpha를 만든다. 합한 four-vector로 invariant mass와 바닥 상태에 대한
excitation energy를 구한다. 전체 beta의 음수로 각 alpha를 boost하여 COM의
상관각·에너지 분율·Dalitz를 계산한다. 재구성 결과에 120도나 에너지 등분배를
강제하지 않는다. 생성 조건을 만족하는 synthetic closure와 실제 재구성 성능을
분리하여 검증한다. 대상 서버 실행과 물리 정확도는 별도 증거가 필요하다.
