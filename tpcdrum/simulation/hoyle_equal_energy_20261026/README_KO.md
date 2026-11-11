# Hoyle 등분배 direct decay MC

`NPS::BeamReaction`이 `C12_3alpha.reaction`에 따라 72 MeV 탄소 빔에서 반동 알파와 7.65 MeV 여기 탄소를 생성한다. `HOYLE_EQUAL_ENERGY=1`이면 TPC 영역에서 여기 탄소를 세 알파로 붕괴시킨다. 부모의 정지계에서 세 알파의 운동에너지는 같고, 세 쌍의 벌어짐 각도는 120°다. 이 모델은 등분배 신호를 생성한다. 환경 변수를 생략하거나 `0`으로 주면 `NPS::Decay`가 반응 파일의 붕괴 설정을 사용한다.

- `HoyleEqualEnergy.hh`는 정지계 운동량과 실험실계 부스트를 계산한다.
- `G4HoyleEqualEnergyModel.hh`는 Geant4에서 여기 탄소를 판별하고 알파 세 개를 secondary track으로 등록한다.
- `tpcdrum/geant4/TPCDrumConstruction.cpp`의 모델 선택부가 환경 변수에 따라 모델을 등록한다.
- `prepare_smoke.py`는 기존 `SimParameters.mac`에서 검출기 설정을 가져와 배치 실행용 `.mac`를 생성한다.

## 빌드와 실행

ROOT, Geant4, NPTool 환경을 설정하고, `tpcdrum`과 `BUILD_NPTOOL`로 구성된 LILAK 저장소의 루트에서 실행한다.

```bash
cmake --build build --target run_sim.exe -- -j4
```

새로 구성할 때 `meta/build_options.cmake`의 `LILAK_PROJECT_LIST`에 `tpcdrum`이 들어 있어야 한다. NPLib 목록은 서버에 설치된 모듈을 선택한다. 이 시뮬레이션의 빌드 설정은 기준 `tpcdrum/CMakeLists.txt`에 있는 `GETDecoder` 하위 경로를 사용한다.

`C12_3alpha.reaction`의 `flat.txt CSR`에는 기존 NPTool의 각분포 입력을 사용한다. `flat.txt`는 NPTool 설치의 `Inputs/CrossSection/`이나 `tpcdrum/simulation/`에 둔다. 기존 NPTool 출력 경로에 다른 작업의 `Simulation.root`가 있다면, 작업별로 분리한 NPTool 실행 환경을 사용한다.

```bash
N_EVENTS=1000000

if python3 tpcdrum/simulation/hoyle_equal_energy_20261026/prepare_smoke.py \
    --repo "$PWD" --events "$N_EVENTS"
then
    cd tpcdrum/simulation
    nohup env HOYLE_EQUAL_ENERGY=1 \
        ./run_sim.exe "SimParameters_DDE_${N_EVENTS}.mac" \
        > "C12_72MeV_DDE_test${N_EVENTS}.log" 2>&1 &
fi
```

`N_EVENTS`는 입사 빔 이벤트 수다. 위 설정의 LILAK 출력은 `tpcdrum/simulation/C12_72MeV_DDE_test1000000.root`에 저장된다. 배치 설정이나 ROOT 출력이 이미 있으면 준비 스크립트가 실행을 중단한다.

```bash
tail -f C12_72MeV_DDE_test1000000.log
```
