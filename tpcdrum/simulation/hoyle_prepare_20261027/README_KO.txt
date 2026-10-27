Hoyle direct MC 준비 — MC 분리와 digitization을 따로 실행
2026-10-27

파일
  C12_make_hoyle_subsets.C     : 원본 MC를 한 번 읽어 3C1/3C2/3C3 MC 생성
  C12_digitize_hoyle_subset.C : 선택한 MC 파일 한 개를 digitization

1. 서버에 설치
이 폴더를 기존 프로젝트 tpcdrum/simulation/ 안에 둔다.
기존 makeSimData.C, simulation 설정, task 또는 detector 소스를 수정하지 않는다.
MC 분리 매크로는 LKMCTrack/LKMCStep dictionary만 필요하며 새 task 빌드가 필요 없다.
digitization은 기존 STDSiEdepSumTask/STHoyleMetaCopyTask를 포함해 빌드된 LILAK이 필요하다.
예전 작업에서 이미 사용하던 라이브러리를 rootlogon.C로 로드한다.
매 실행은 새 ROOT 세션으로 하며 + 또는 ++를 붙이지 않는다.
두 .C 매크로에는 #include 지시문이 없다.

2. 먼저 MC 분리만 실행
새 원본 파일이 C12_72MeV_DDE_test1000000.root라는 예시다.
입력 이름은 실제 파일명으로 바꾼다. outputPrefix는 경로를 포함해도 된다.

cd ~/workspace/lilak_seunghwan/tpcdrum/simulation
root -l -b -q ../../macros/rootlogon.C \
  'hoyle_prepare_20261027/C12_make_hoyle_subsets.C("C12_72MeV_DDE_test1000000.root","C12_DDE_allcomb")'

출력(현재 simulation 폴더):
  C12_DDE_allcomb_3C3.root : 각 direct decay의 alpha a,b,c 모두
  C12_DDE_allcomb_3C2.root : 각 direct decay에서 (a,b),(a,c),(b,c)
  C12_DDE_allcomb_3C1.root : 각 direct decay에서 (a),(b),(c)

확인된 direct decay N개라면 출력 event entries는 N,3N,3N이다.
3C1/2/3은 선택한 붕괴 alpha 수이며 Si 도달 multiplicity가 아니다.
Si 도달 여부, TPC 신호 유무, layer 수, theta, DDE 등분배 조건으로 선별하지 않는다.
따라서 TPC에 신호를 남기지 않은 alpha도 해당 조합에 보존된다.
출력 prefix를 생략하면 원본 이름 뒤 _decay가 붙는다.
기존 출력 파일은 덮어쓰지 않는다. 출력 폴더는 먼저 존재해야 한다.

분리 함수 인자:
  C12_make_hoyle_subsets(inputMC, outputPrefix="", maxEntries=-1,
                         vertexToleranceMM=0.001)
- maxEntries=-1: 원본 전체 읽기. 양수: 원본 앞부분만 읽는 시험.
- 시험 예: 별도 prefix로 (...,"C12_DDE_trial",1000).
- maxEntries는 원본 beam event 수이며 선택된 붕괴 N을 지정하는 값이 아니다.
- 반환값 true는 N>0인 정상 생성, false면 Error 로그를 확인한다.
  ROOT가 bool 반환값을 OS 종료 코드로 바꾸는 것으로 가정하지 않는다.

3. 선별과 recoil 제외
- 저장된 MCTrack의 ParentID로 alpha들을 묶는다.
- 실제 저장된 부모의 PDG가 C12이고 그 직접 alpha 딸이 정확히3개여야 한다.
  C12의 nuclear PDG 마지막 excitation/isomer digit도 허용한다.
- 세 딸 alpha 생성점이 모든 쌍에서 0.001mm 이내인지 확인한다.
- 후보0개/후보여러개/중복 또는 잘못된 track ID는 별도 집계하여 제외한다.
- recoil alpha는 그 C12 붕괴 딸이 아니므로 제외한다.
  같은 vertex나 비슷한 theta라는 이유만으로 딸이라고 판정하지 않는다.
- TrackID 번호를 고정하지 않는다. 부모 관계가 없으면 추측하지 않는다.
- 이 선택은 direct C12->3alpha topology 확인이다. 불변질량7.65MeV,
  alpha의 COM 에너지 등분배 또는120도 자체를 재검증하는 선택은 아니다.
  sequential C12->alpha+Be8->3alpha는 직접딸3개 조건을 통과하지 않는다.

4. 보존하는 정보
선택한 alpha의 모든 저장된 track vertex/위치/운동량/운동에너지/volume/status.
선택한 alpha TrackID에 직접 속한 모든 MCStep* branch의 x,y,z,time,Edep.
TPC와 Si뿐 아니라 저장된 다른 MCStep branch도 같은 방식으로 보존한다.
부모 탄소/recoil alpha/선택하지 않은 alpha/secondary 자손의 별도 step은 제외한다.
이 정책은 기존 alpha-own-step subset 정책과 같다.
출력의 alpha ID는1..K, ParentID=0으로 바뀌고 step ID도 함께 바뀐다.
원래 정보를 지우는 대신 HoyleMeta에 별도 보존한다:
  원본 entry, 원래 C12 부모ID, 선택 alpha의 원래ID/부모,
  전체triplet 원래ID, 선택index, 조합index, 붕괴위치.
HoyleMeta는 출력 event와 정확히1:1 대응한다.
원본 파일 경로와 UUID, 선택 방식, 읽은 원본 수/수용 수/거절 수를 저장한다.
event에는 MCTrack/MCStep*만 기록하고 원본의 top-level non-tree parameter를 보존한다.

원본 entry가 같은 형제 조합들은 독립 train/test 표본이 아니다.
학습 split은 HoyleSourceUUID+HoyleSourceEntry를 묶어 나눠야 한다.

5. 완료 확인
각 출력 파일의 다음 객체를 확인할 수 있다.
  SubsetComplete=1       : 요청한 범위 읽기와 출력 완료
  SubsetInputEntries     : 원본 전체 event 수
  SubsetRequestedEntries : 이번에 읽기로 한 event 수
  SubsetInspected        : 실제 읽은 event 수
  SubsetProcessedAll=1   : 원본 전체 읽음(부분 시험이면0)
  SubsetAccepted         : 수용한 direct decay N
  SubsetNoTriplet / SubsetAmbiguous / SubsetMalformed
N=0이면 empty output을 그대로 남기고 false를 반환하며 digitization은 시작하지 않는다.
예외/디스크오류가 발생한 출력은 SubsetComplete=1이 없으므로 사용하지 않는다.

6. 이후 별도로 digitization(지금 MC 분리만 할 때는 실행하지 않음)
현재 simulation 디렉토리에서 한 번에 subset 파일 하나를 처리한다.
3C1/3C2/3C3은 같은 매크로에 입력만 바꾸면 된다.

먼저3C3 앞10개 시험:
root -l -b -q ../../macros/rootlogon.C \
  'hoyle_prepare_20261027/C12_digitize_hoyle_subset.C("C12_DDE_allcomb_3C3.root","../../data/C12_DDE_allcomb_3C3_digi_SiEdep_first10.root",10)'

전체3C3:
root -l -b -q ../../macros/rootlogon.C \
  'hoyle_prepare_20261027/C12_digitize_hoyle_subset.C("C12_DDE_allcomb_3C3.root","../../data/C12_DDE_allcomb_3C3_digi_SiEdep.root")'

전체3C2:
root -l -b -q ../../macros/rootlogon.C \
  'hoyle_prepare_20261027/C12_digitize_hoyle_subset.C("C12_DDE_allcomb_3C2.root","../../data/C12_DDE_allcomb_3C2_digi_SiEdep.root")'

전체3C1:
root -l -b -q ../../macros/rootlogon.C \
  'hoyle_prepare_20261027/C12_digitize_hoyle_subset.C("C12_DDE_allcomb_3C1.root","../../data/C12_DDE_allcomb_3C1_digi_SiEdep.root")'

digitization 함수 인자:
  C12_digitize_hoyle_subset(inputSubset, outputDigi="", numEvents=-1,
                             startEntry=0, parameterFile="SimParameters.mac")
- numEvents=-1: startEntry 이후 전부. 양수: 그 개수(남은 개수로 제한).
- startEntry는 원본 beam event가 아닌 해당 subset 파일의 entry다.
- 구간을 나누려면 서로 다른 outputDigi 이름으로 numEvents/startEntry를 지정한다.
  기존 파일에 이어 붙이지 않는다. 같은 입력의 같은 구간을 다시 쓰지 않도록 관리한다.
- outputDigi 생략 시 입력명 뒤 _digi_SiEdep.root를 붙인다.
- 실제 MC를 만들 때 쓴 SimParameters_DDE_1000000.mac 등을 마지막 인자로 지정 가능.
- gas/drift/GEM gain/electronics 설정을 이 매크로가 수정하지 않는다.
- 원본 생성기 HOYLE_EQUAL_ENERGY 환경변수는 digitization에 다시 필요하지 않다.

순서: STDSiEdepSumTask -> STDDriftElectronMaker -> STDElectronicsMaker
      -> STHoyleMetaCopyTask.
TPC RawPad/MCTagTPCDrum 및 Si deposited energy(MeV)를 저장한다.
Si 파형을 만드는 STDSiResponseMaker는 이 경로에 넣지 않는다.
HoyleMeta와 원본 UUID/선택 proof를 digi에도 전달한다.
입력은 이 폴더의 새 분리 매크로가 만든 completed subset으로 제한한다.
기존 ROOT 자동 파일 분할로 event/metadata가 어긋나는 것을 피하도록
자동 rollover 한도를 높인다. 큰 작업은 numEvents/startEntry로 명시적으로 나눌 수 있다.
C12DigiComplete=1은 요청한 구간이 event/metadata 정렬 상태로 저장된 경우에만 기록한다.
SubsetProcessedAll은 MC 분리의 전체처리 여부이며 digi 전체처리 여부가 아니다.
digi의 처리범위는 C12DigiInputEntries/StartEntry/RequestedEvents로 확인한다.

7. 검증 범위
소규모 ROOT fixture로 MC 선별, 모든 조합, recoil 제거, track/step/metadata 보존,
부분 범위, 기존 파일 보호를 확인한다(상세 VALIDATION.json 참조).
fixture의 LKMCTrack/LKMCStep은 테스트용 API 대체 클래스다.
실제 LILAK/Garfield의 물리 digitization까지 로컬에서 돌렸다는 뜻은 아니다.
실제 TPC 파형 생성은 서버의 기존 LILAK 환경에서 짧은 시험부터 확인한다.
