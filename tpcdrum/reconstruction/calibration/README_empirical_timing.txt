MC timing 연구 도구

이 도구는 reconstruction inference와 분리되어 있다. 저장된 MC 경로를 정답으로
사용하여 관측 waveform의 시간 통계량과 비교하는 연구용이다. 계산한 affine map이
실제 검출기 또는 native transport의 보정 완료를 뜻하지 않는다.

  python -m reconstruction.calibration.study_empirical_timing --help

export 단계
  --stage export --manifest frozen_selection.json --output NEW_EXPORT_DIR
  ROOT 실행 파일은 --root-executable로 지정할 수 있다.
  manifest는 파형·경로를 보기 전에 고정한 입력 목록, source-event 분할,
  선택 CSV와 SHA-256, 이벤트/source 수를 요구한다.
  CSV 필드는 request_index,input_file_index,entry,source_entry,split,path,
  vertex_x_mm,vertex_y_mm,vertex_z_mm다. vertex는 MC 원래 좌표이며 정답 정합성
  검사에만 사용한다. 같은 source의 sibling은 한 분할에 둔다.
  서로 다른 파일의 source 번호는 공통 lineage를 먼저 확인해야 한다.
  configuration_frozen_before_waveforms_and_paths=true 선언만으로 그 확인을
  대신할 수 없다. 이 모듈은 그 외부 감사를 자동으로 수행하지 않는다.
  관측 통계량과 MC 정답은 각각 timing_observations.root, timing_truth.root에 저장한다.

fit 단계
  --stage fit --observations-root timing_observations.root \
    --truth-root timing_truth.root --output NEW_FIT_DIR
  원본 MC 파일 없이 위 두 ROOT만으로 재분석한다.
  계수는 training partition만으로 결정하고 validation/test에는 같은 계수를 적용한다.
  source-event 전체를 묶어 재표집하고, 누락·중복 교차·관측되지 않은 레이어는
  coverage에 남긴다. waveform 창의 끝에서 신호가 잘린 경우를 확인해야 한다.
  출력은 production의 calibrated_for_charge_mean 플래그를 켜지 않는다.

저장된 step 좌표를 다시 변환하지 않는다. 이 코드의 native frame 규약과
사용자 데이터의 detector geometry가 같은지 먼저 확인해야 한다.
새 기하·가스 조건이나 장치 보정으로 자동 일반화하지 않는다.
