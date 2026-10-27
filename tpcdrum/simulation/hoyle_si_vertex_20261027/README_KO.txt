원본 direct-decay MC의 Si alpha multiplicity vs vertex
2026-10-27

실행 (기존 LILAK dictionary가 로드되는 ROOT 환경)

cd ~/workspace/lilak_seunghwan/tpcdrum/simulation
root -l -b -q ../../macros/rootlogon.C \
  'hoyle_si_vertex_20261027/C12_study_hoyle_si_by_vertex.C("C12_72MeV_DDE_test1000000.root")'

파일명은 실제 원본 transport MC 이름으로 바꾼다.
3C1/3C2/3C3로 잘라낸 subset이나 RawPad digitization 파일은 입력이 아니다.
매크로에는 #include가 없으며 +/++ 없이 해석 실행한다.
새 task 설치나 프로젝트 재빌드가 필요하지 않다.
파일 경로는 절대경로 또는 현재 simulation 폴더 기준 상대경로를 쓸 수 있다.

인자
  C12_study_hoyle_si_by_vertex(inputMC="C12_72MeV.root",
                               siThresholdMeV=0.1, maxEvents=-1)
  -1: 전체 원본 event를 읽음.
  양수: 원본 앞부분 그 개수만 읽음(선택된 붕괴 개수 지정이 아님).
  시험 예: (..."C12_72MeV_DDE_test1000000.root",0.1,10000).

두 가지 multiplicity를 동시에 출력
1. arrival (주 결과)
   선택된 붕괴 alpha 자신의 MCStep가 8개 Si 중 하나 이상에 저장되면 도달로 센다.
   같은 alpha의 여러 step/여러 Si 방문은 한 alpha로 센다.
   0-Edep step도 기록돼 있다면 arrival에 포함한다.
2. edep (보조 비교)
   그 alpha의 8개 Si own-track deposit 합이 threshold보다 엄격히 큰 경우에 센다.
   기본은 >0.1 MeV. ==0.1 MeV는 통과하지 않는다.

N=0,1,2,3은 Si에 도달한 붕괴 alpha 개수이다.
서로 다른 Si module이 몇 개 hit됐는지를 뜻하지 않는다.
예: alpha3개가 같은 Si에 모두 도달하면 N=3이다.
이는 저장된 MC step으로 정의한 도달/에너지 통계이며 ADC/trigger/reco 효율이 아니다.
0-Edep step이 저장되지 않는 MC에서는 arrival을 정확한 모든 경계 통과 수로 해석하면 안 된다.

붕괴 선택과 분모
- ParentID로 직접 C12->3alpha genealogy를 확인한다.
- 실제 저장된 C12 부모, 정확히3개 직접 alpha 딸, 공통 생성점(0.001mm 이내)을 요구한다.
- alpha ID를 고정하지 않으므로 4/5/6이라는 번호에 의존하지 않는다.
- recoil alpha와 secondary 자손은 세지 않는다.
- Be8 경유 sequential decay는 이 매크로의 선택 대상이 아니다.
- 후보 없음/여러 후보/잘못된 track ID는 별도 상태로 집계한다.
- 이것은 direct topology 판정이다. Hoyle 여기 에너지나 DDE 등분배의 독립 검증은 아니다.
- 분모는 해당 vertex bin의 선택된 direct decay 전체이며 N=0도 포함한다.
- Si 도달, TPC hit, layer 수, theta, 에너지, transverse vertex로 표본을 자르지 않는다.

좌표와 bin
  u_pad=-X_MC, v_pad=Z_MC+66, w_pad=Y_MC+128.8 [mm].
  딸 alpha의 생성 위치를 붕괴 vertex로 사용한다. 입사 C12의 최초 위치를 쓰지 않는다.
  그래프는 v_pad에 대한 분포다. 기본25bins, -290..210mm, bin폭20mm.
  범위 밖 이벤트는 ROOT flow bins 및 CSV/전체 통계에 보존한다.
  빈 bin은 가짜0% 점을 넣지 않고 확률 그래프에서 생략(CSV nan)한다.

구간별 보조 통계 (v축에 대한 명목 reference 구간)
  before_pad:      v < -6
  pad_v_interval: -6 <= v < 138
  pad_to_Si:      138 <= v < 199
  at_or_after_Si: 199 <= v
  v=199mm는 현재 snapshot의 Si 중심평면이다(1mm 두께의 전면은198.5mm).
  이 구간이나 audit의 in_pad_reference_box는 실제 active-volume cut이 아니다.
  legacy reference box는 u=[-36.95,36.95),v=[-6,138),w=[0,128.8)이며
  TPCDrum::IsInBoundary()와 다르다. 어떤 selection에도 사용하지 않는다.
  detector geometry가 바뀌면 좌표/구간 상수도 그 geometry에 맞춰 재확인한다.

각 canvas의 네 패널
  1. vertex별 N=0/1/2/3 누적 이벤트 수
  2. exact multiplicity P(N=k | selected decay in bin)
  3. event acceptance P(N>=k | selected decay in bin)
  4. 위 reference 구간별 exact multiplicity
arrival/edep 각각 canvas를 만든다.
오차막대는 각 event 확률의 68.27% Clopper-Pearson 구간이다.
세 sibling alpha를 3개의 독립 event처럼 취급하지 않는다.

출력: 현재 실행 디렉토리에 inputStem_hoyle_si_vtx_UUID.*
  .root        : 두 canvas, graph/histogram, event_audit와 상태 집계
  .csv         : 전체/구간/v-bin별 N0..N3, 확률과 구간, 평균 multiplicity
  _arrival.svg : arrival canvas (벡터 그림)
  _edep.svg    : edep canvas
  .pdf         : 위 두 canvas의 2페이지 PDF
UUID 접미사로 기존 결과와 겹치지 않게 저장한다. 원본 MC는 READ 모드로만 연다.
입력 UUID, 원본 event entry, 부모/alpha ID, 좌표, Si mask/Edep를 ROOT에 기록한다.
event_audit status=0: 선택 가능한 triplet 없음(물리적으로 붕괴하지 않았다는 증명은 아님),
status=1: 선택, status=2: 여러 triplet, status=3: malformed MCTrack.
analysis_complete=1은 요청 범위를 정상 분석한 경우 기록한다.
선택된 붕괴가0이면 해석할 효율 표본이 없다는 경고를 출력한다.

GitHub에서 기존 서버 파일 변경 없이 받기

git clone --depth 1 --single-branch --branch codex/hoyle-si-vertex-study \
  https://github.com/dydtjrl08/lilak.git \
  ~/workspace/lilak_si_vertex_delivery_20261027

bash ~/workspace/lilak_si_vertex_delivery_20261027/tpcdrum/simulation/hoyle_si_vertex_20261027/install_new_macro.sh \
  ~/workspace/lilak_seunghwan

설치 script는 새 macro 폴더만 복사한다.
목적지 폴더가 있으면 중단하며 기존 파일/설정/task/data/.git을 바꾸지 않는다.
기존 checkout에서 pull/reset/clean을 실행하지 않는다.
설치 후 위의 ROOT 실행 명령을 사용한다.

검증 범위
실제 ROOT I/O와 테스트용 LKMCTrack/LKMCStep 인터페이스를 사용한 합성 MC로
recoil/중복 counting/threshold/분모/좌표구간/flow/CSV/graph/원본 보존을 확인한다.
사용자의 새 원본 MC나 서버 ROOT6.24에서 직접 실행한 결과는 아니다.
