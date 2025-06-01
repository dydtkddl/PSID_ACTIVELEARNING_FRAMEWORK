```markdown
# PSID Active Learning Framework

## 1. 전체 목적
MOF(Materials–Organic Framework)의 흡착 등온선(Isotherm) 파라미터를 예측하기 위해, 풀 GCMC(Grand Canonical Monte Carlo) 시뮬레이션 대신 일부 저압 조건에서만 GCMC를 수행하고 나머지 포인트를 머신러닝 모델로 예측하는 파이프라인을 제공합니다.  
이를 통해 다음을 달성합니다:
- 수많은 MOF 후보에 대해 실험 또는 풀 GCMC를 수행하지 않아도 되므로, 시간과 계산 비용을 크게 절감  
- 저압 흡착량을 물리적으로 의미 있는 Feature로 활용하여, 고압 등온선 예측 정확도 확보  
- Monte Carlo Dropout 기반 Active Learning으로 불확실성이 큰 데이터만 추가 라벨링하여, 최소한의 GCMC 샘플링으로 모델 성능 극대화  

---

## 2. 배경 및 필요성
- **MOF Isotherm 파라미터 중요성**  
  MOF의 흡착 성능을 정량화하려면 등온선 파라미터(예: Langmuir, BET 등)가 필요합니다.  
- **풀 GCMC 한계**  
  MOF 후보가 수천 개 이상일 때, 다양한 압력·온도 포인트에서 GCMC를 모두 수행하는 것은 현실적으로 불가능할 만큼 시간이 오래 걸립니다.  
- **저압 흡착량 활용**  
  저압 영역에서 측정되는 흡착량은 MOF-가스 초기 상호작용을 반영하며, 고압 등온선과도 높은 상관관계를 가집니다.  
- **Active Learning 도입**  
  불확실성이 큰 데이터 포인트만 골라 추가 GCMC 샘플링을 수행하면, 전체 라벨링 비용을 최소화하면서 ML 모델의 예측 신뢰도를 높일 수 있습니다.  

---

## 3. 주요 기능
1. **MOF 스크리닝**  
   - 벤치마크 MOF 데이터베이스에서 특정 가스의 기공 직경(PLD)을 기준으로 타겟 MOF 후보군을 선별하고, `target_mofs.csv` 파일을 생성합니다.  
2. **저압 흡착량 GCMC 샘플링**  
   - RASPA와 연동하여, 다수 노드를 활용한 병렬 Low-pressure GCMC를 실행하고 흡착량 데이터를 수집합니다.  
3. **Active Learning(추후 구현 예정)**  
   - Monte Carlo Dropout 기반 불확실성 추정으로, Epistemic Uncertainty가 높은 지점을 선별해 추가 GCMC 라벨링 수행  
   - 반복 학습 루프를 통해 예측 정확도 점진적으로 개선  

---

## 4. 폴더 구조 예시
```

PSID\_ACTIVELEARNING\_FRAMEWORK/
├── config.json
├── create\_isotherm\_project.py
├── create\_low\_pressure\_gcmc.py
├── ActiveLearning/            # (추후 개발) Active Learning 관련 스크립트
├── utils/                     # 데이터 전처리 및 공통 유틸리티 함수
└── README.md                  # 본 문서

```

---

## 5. 필요 조건
- Python 3.7 이상  
- RASPA 2.x 설치 (Low-pressure 및 Full GCMC 시뮬레이션용)  
- 주요 Python 라이브러리:  
```

numpy
pandas
scikit-learn
torch
matplotlib

````
- 병렬 실행 환경(SSH key 기반 무비밀번호 접속이 설정된 클러스터 노드)

---

## 6. 설치 및 환경 설정
1. **저장소 복제 및 디렉터리 이동**  
 ```bash
 git clone https://github.com/your-username/PSID_ACTIVELEARNING_FRAMEWORK.git
 cd PSID_ACTIVELEARNING_FRAMEWORK
````

2. **가상환경 생성(권장)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **필요 라이브러리 설치**

   ```bash
   pip install numpy pandas scikit-learn torch matplotlib
   ```

4. **RASPA 실행 경로 확인**

   * 시스템에 맞게 `RASPA_PATH`를 확인합니다. (예: `/usr/local/raspa`)

---

## 7. 구성 파일(config.json) 설정 가이드

`config.json`에는 크게 세 가지 영역을 설정해야 합니다. 실제 예시는 아래처럼 최소한의 항목만 남긴 템플릿을 참고하세요.

```jsonc
{
  "mof_database_path": "mof_database/2019-11-01-ASR-public_12020.csv",

  "kinetic_diameters": [
    // 원하는 가스만 최소한으로 남기세요.
    {"name": "Hydrogen", "formula": "H2", "molecular_mass": 2, "kinetic_diameter_A": 2.89},
    {"name": "Carbon dioxide", "formula": "CO2", "molecular_mass": 44, "kinetic_diameter_A": 3.30}
  ],

  "main_config": {
    "main": {
      "gas": null,                 // create_isotherm_project.py 실행 시 자동 채워짐
      "temperature": null,         // 동일하게 자동 채워짐
      "target_mofs_csv": "./target_mofs.csv"
    },

    "GCMC": {
      "NumberOfCycles": 20000,       // GCMC 총 사이클 수
      "NumberOfInitializationCycles": 10000, // 초기 사이클
      "PrintEvery": 1000,            // 로그 출력 주기

      "UseChargesFromCIFFile": null, // CIF 파일에서 전하 정보 사용 여부
      "ExternalTemperature": null,   // create_isotherm_project.py에서 채워짐
      "ExternalPressure": null,      // create_isotherm_project.py에서 채워짐
      "Forcefield": "GarciaPerez2006ForceField",

      "GAS": null,                   // create_isotherm_project.py에서 채워짐
      "MoleculeDefinition": "ExampleDefinitions",

      "CUTOFFVDW": 14,
      "CUTOFFCHARGECHARGE": 14,
      "CUTOFFCHARGEBONDDIPOLE": 14,
      "CUTOFFBONDDIPOLEBONDDIPOLE": 14
    },

    "Low_Pressure_GCMC": {
      "Parallel": {
        "nodes": ["node01", "node02"],   // 병렬 실행할 노드 호스트명
        "max_cpu_fraction": 0.8          // 노드당 최대 CPU 사용 비율 (0~1)
      }
    },

    "Active_GCMC": {
      "initial_sampling": {
        "initial_fraction": 0.01,     // 초기 샘플링 비율 (예: 전체 MOF 중 1%)
        "Parallel": {
          "nodes": ["node01", "node02"],
          "max_cpu_fraction": 0.8
        }
      },
      "active_sampling": {
        "target_fraction": 0.1,       // 최종 목표 샘플링 비율 (예: 10%)
        "n_samples": 10,              // 매 라운드마다 선택할 GCMC 포인트 수
        "Parallel": {
          "nodes": ["node01", "node02"],
          "max_cpu_fraction": 0.8
        },
        "neural_network": {
          "model_spec": {
            "hidden_layers": [
              {"hidden_dim": 64, "dropout": 0.1, "activation_func": "ReLU"},
              {"hidden_dim": 64, "dropout": 0.1, "activation_func": "ReLU"}
            ]
          },
          "dataset": {
            "BATCH_SIZE": 64
          },
          "training": {
            "max_epoch": 500,
            "patience": 30,
            "learning_rate": 1e-3
          },
          "prediction": {
            "mcd_numbers": 20        // Monte Carlo Dropout 횟수
          }
        }
      }
    }
  }
}
```

* **`mof_database_path`**

  * 벤치마크 MOF 목록 CSV 파일 경로를 지정합니다.
* **`kinetic_diameters`**

  * 자주 사용할 가스만 최소한으로 남겨 두세요. 나머지는 기본값으로 자동 포함됩니다.
* **`main_config.main`**

  * `gas`, `temperature` 필드는 `create_isotherm_project.py` 실행 시 자동으로 채워집니다.
  * `target_mofs_csv`는 생성될 MOF 리스트 파일 경로입니다.
* **`main_config.GCMC`**

  * 기본 GCMC 시뮬레이션 파라미터를 정의합니다.
  * `Forcefield`, 컷오프 값, 사이클 수 등을 프로젝트에 맞게 조정하세요.
* **`main_config.Low_Pressure_GCMC.Parallel`**

  * 병렬 노드(hostname)와 CPU 사용 비율 설정만 입력하면 됩니다.
* **`main_config.Active_GCMC`**

  * Active Learning 라운드별 샘플링 비율과 노드 설정을 지정합니다.
  * Neural Network 구조, 학습률, 드롭아웃 등은 추후 필요 시 조정하세요.

---

## 8. 사용 방법

### 1) 프로젝트 생성

원하는 **기체 이름**(예: `Ar`, `CO2`, `CH4`)과 \*\*온도(단위: K)\*\*를 지정하여 프로젝트 디렉터리를 생성합니다.

```bash
python create_isotherm_project.py <기체이름> <온도>
```

* 예:

  ```bash
  python create_isotherm_project.py CO2 313.0
  ```
* 동작 요약:

  1. `config.json`의 `mof_database_path`에서 MOF 목록을 로드
  2. 지정한 기체의 PLD 기준으로 타겟 MOF 후보를 필터링
  3. 새로운 폴더(예: `Isotherm_CO2_313.0K_20250601_102530/`)를 생성
  4. 해당 폴더에 `target_mofs.csv` 파일 생성

### 2) 저압 흡착량 GCMC 샘플링

생성된 프로젝트 디렉터리로 이동 후, 전체 타겟 MOF에 대해 저압 GCMC를 실행합니다.

```bash
cd Isotherm_CO2_313.0K_20250601_102530
python create_low_pressure_gcmc.py
```

* 동작 요약:

  1. `target_mofs.csv`에서 MOF ID와 구조 정보를 읽음
  2. `config.json`의 `Low_Pressure_GCMC` 설정에 따라 RASPA 시뮬레이션 입력 파일 생성
  3. 병렬 노드(`nodes`)를 활용해 다수 노드에서 Low-pressure GCMC 실행
  4. 각 MOF별 시뮬레이션 결과(흡착량, 로그 파일 등)를 폴더에 저장
  5. 최종적으로 `low_pressure_uptake.csv`에 모든 MOF의 흡착량 데이터를 정리

### 3) Active Learning 프로세스 진행 (추후 구현 예정)

Active Learning 모듈이 완성되면, 다음 단계로 진행합니다:

1. **초기 샘플링**

   * `initial_fraction` 비율 만큼 무작위로 선택한 MOF에 대해 GCMC 라벨링 수행
2. **모델 학습**

   * 저압 흡착량 데이터를 기반으로 Neural Network 모델 학습
   * Monte Carlo Dropout을 적용해 데이터 불확실성(에피스테믹 불확실성) 계산
3. **불확실성 기반 샘플링**

   * `n_samples` 개수만큼 불확실성이 높은 포인트를 선별하여 추가 GCMC 라벨링
4. **반복 학습**

   * 새로 라벨링된 데이터를 추가해 모델 업데이트
   * `target_fraction`에 도달할 때까지 2\~4를 반복

---

## 9. 예제 실행 흐름

1. **`config.json` 수정**

   * 벤치마크 MOF 데이터베이스 경로(`mof_database_path`) 확인
   * 병렬 노드 목록(`Low_Pressure_GCMC.Parallel.nodes`) 설정
2. **프로젝트 생성**

   ```bash
   python create_isotherm_project.py Ar 298.0
   ```

   * 결과 폴더: `Isotherm_Ar_298.0K_20250601_110015/`
   * `target_mofs.csv` 생성
3. **저압 GCMC 실행**

   ```bash
   cd Isotherm_Ar_298.0K_20250601_110015
   python create_low_pressure_gcmc.py
   ```

   * RASPA 입력/출력 자동 생성
   * `low_pressure_uptake.csv`에 MOF별 흡착량 기록
4. **(추후) Active Learning 실행**

   ```bash
   cd ActiveLearning
   python run_active_learning.py
   ```

   * 초기 샘플링, 모델 학습, 불확실성 기반 샘플링, 반복 업데이트

---

## 10. 향후 계획

* Active Learning 모듈 완성 및 통합
* 학습 성능 평가 및 시각화 (R², MAE 등)
* 다중 기체·온도 배치 처리 지원
* 결과 모니터링을 위한 Web GUI 개발

---

## 11. 라이선스


```
```
