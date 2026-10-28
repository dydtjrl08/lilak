Sequential track finding / common-vertex study — 2026-10-27

기본 순서
  깨끗한 구간의 첫 track seed -> soft peel -> 두 번째 track 후보 여러 개 ->
  첫 관측 구간의 국소 상태로 vertex 후보 영역 -> 세 번째 track 가설 ->
  원래 관측 waveform에서 K1/K2/K3 가설 비교 -> 독립 track fitting ->
  같은 원본 측정값으로 free/common-vertex 공동 피팅 비교.

현재 config.json의 기본값은 finding.method="sequential_vertex",
vertex.method="joint_tracks"다. 이전 hough/ransac/tls/cnn/cnn_hough/
waveform_hypotheses 및 단순 DCA vertex도 선택할 수 있다.

전하를 지우는 방식
  확실한 설명 성분의 일부만 confidence×peel_fraction만큼 뺀다.
  peel_fraction은 반드시 0보다 크고 1보다 작다. 공유 pad 전체를 삭제하지 않는다.
  residual은 다음 seed를 찾는 보조 자료이며 최종 적합 대상은 원래 파형이다.
  각 가설의 geometry/template/amplitude는 heldout pad를 제외해서 적합한다.
  그 후 원래 heldout 이웃 pad의 예측 오차로 가설을 비교한다.
  vertex region은 관측에서 얻은 proposal/gate이며 독립적인 추가 측정이 아니다.

Multiplicity와 예외
  최대 track 수 3은 정답으로 강제하는 개수가 아니다. K1/K2/K3를 비교한다.
  깨끗한 seed가 없으면 전역 waveform 가설로 되돌아가며 fallback 이유를 저장한다.
  동일 궤적의 time-tail 조각이나 중복 fragment를 별도 입자로 세지 않도록 검사한다.
  완전 중첩, 열린 vertex 영역, ill-conditioned mixture, 응답 모델 불일치는 ambiguous다.
  K3는 가장 좋은 낮은 multiplicity 가설보다 정해진 heldout 개선을 얻어야 선택된다.
  분리되었다고 선언하는 것과 MC 정답에 가까운 분배를 우연히 얻는 것을 구분한다.

Vertex
  첫 hit layer는 생성점 좌표가 아니라 탐색 방향 단서다. layer extent는 soft support로만 쓴다.
  vertex_v_scan_mm은 명시적 수치 탐색 범위다. 실제 반응 영역이라고 간주하지 않는다.
  실제 reaction_volume을 source와 함께 지정한 경우 바깥 candidate는 기각하되 위치는 보존한다.
  평행·근평행에서는 extended_region을 남기고 정확한 한 점을 만들어내지 않는다.
  3D는 drift velocity/t0/electronics-delay가 보정된 경우에만 사용한다.
  보정이 없으면 u-v 진단만 저장하며 physical vertex_valid는 0이다.

공동 피팅
  원래 layer 측정값과 covariance를 한 번만 사용한다. 이전 CNN/Kalman의 fitted point나
  같은 track에서 계산한 vertex를 새 독립 측정으로 다시 넣지 않는다.
  free fit과 joint fit은 동일한 측정·과정잡음 모델을 사용한다.
  process_noise_mode="none"은 직선 GLS, "tuned"+angular_variance_rad2_per_mm는
  angular-kick piecewise 모델이다. 이 공동 피팅을 full Kalman이라고 부르지 않는다.
  기존 EKF/UKF/RTS 에너지-aware 단계는 별도로 유지한다.
  2D 자유 직선 두 개에는 common vertex 조건만으로 각도 정보가 늘지 않는 검사가 있다.
  외부 beam/vertex prior는 독립적인 관측·covariance·source를 명시해야 한다.
  공유 clock offset은 각 hit의 독립 오차 대신 공통 nuisance로 넣을 수 있다.
  covariance와 profile 폭은 supplied proxy/process 모델에 조건부다. 검출기 resolution 검증은 아니다.
  수렴/가역 행렬만으로 common origin을 승인하지 않는다. 명시 χ² 품질 gate와 실제 영역을 검사한다.

ROOT 출력
  sequential_history와 column/stage code는 record_json에 함께 저장된다.
  seed lines/ranks, 두 clean mask, pair indices/regions, region envelope,
  남은 waveform, 후보 lines/scores를 같은 ROOT에서 읽을 수 있다.
  vertex_free_theta_uv_deg와 vertex_joint_theta_uv_deg는 같은 모델의 전후 비교다.
  vertex_joint_comparison_valid가 0인 실패 표본은 없어지지 않고 coverage에 남는다.
  짧은 seed region과 성능 비교용 전체 free-fit region은 구분한다.
  3D 공동 피팅이 track을 바꾸면 기존 Si 연결·물질 경로·에너지·운동학 결과를 무효화한다.
  pre_vertex_*가 원래 상태를 보존한다. 바뀐 궤적으로 material trace를 다시 실행해야 한다.

각도별 benchmark
  angular_benchmark.benchmark(methods,truth_root,output_root,...).
  CLI: python run.py --stage angular_benchmark --config comparison.json --output comparison.root
  comparison.json의 angular_benchmark에 methods, truth_root, 고정 thresholds를 지정한다.
  methods={"Hough":{"before":"fitting.root","after":"joint_vertex.root"},...} 형식이다.
  최초 단계에서 정한 이벤트 전체 alpha 순열을 재피팅 후에도 고정한다.
  기본 minimum_pair는 이벤트마다 lab u-v 각도 차이가 가장 작은 쌍 하나를 평가한다.
  all_pairs를 쓰면 같은 이벤트의 세 쌍은 상관되어 있다는 점을 별도로 고려한다.
  purity/recall/visible-layer 기준은 점수를 보기 전에 고정한다.
  conditional MC recovery, merge, fake split, theta bias/std/RMSE, vertex error,
  전후 coverage 및 공통 유효 표본의 paired 변화를 기록한다.
  method-accepted recovery는 명시적 non-ambiguous 선언까지 요구한다.
  Hough proximity-only 같은 미검증 결과도 conditional MC recovery는 따로 볼 수 있다.
  작은 각도의 성공 사례만 골라 resolution을 표시하지 않는다.
  원본/model 없이 저장된 benchmark ROOT만으로 표와 PDF/PNG를 다시 만들 수 있다.

현재 실제 자료의 한계
  이전 pilot48 및 entry5와 source group이 겹치지 않는 기존 MC24개를 새 비교 표본으로 고정했다.
  설정·threshold는 score를 보기 전에 저장했다. IEEE 음수 background 수치 오차만 수정해
  같은 설정으로 재실행했고 최초 실패 결과와 사유를 보존했다.
  기존 3C3에는 관측 Si 에너지가 없고 charge-mean drift 보정이 없어 이 비교는 u-v 진단이다.
  새 DDE digi, 실제 보정, native server exporter 실행 후 물리적 3D/질량/COM 검증이 필요하다.
  120도와 에너지 등분배는 finding/vertex 적합 조건으로 쓰지 않는다.
