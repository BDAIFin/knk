# FDS (2-Stage) — Fraud Detection System

온라인 카드 거래 사기 탐지를 위한 2-Stage 의사결정 파이프라인입니다.

본 프로젝트는 다음과 같은 구조를 갖습니다.

- **Stage1**: 거래 단독 기반의 경량 1차 판별  
  → 전체 거래를 빠르게 스코어링하고 Stage2로 넘길 대상(Review ids)을 선별

- **Stage2**: 사용자/카드 컨텍스트 + 히스토리 기반의 고정밀 2차 판별  
  → Stage1에서 선별된 대상에 대해 정밀 판단 수행

- **CHECK (Drift)**: 운영 환경에서의 분포 이동 감지 및 대응 전략 검증

---

# 1. Repository Structure

## 1) EDA&FEATURE

분석과 피처 의사결정을 담당하는 영역입니다.  
모델링 이전 단계에서 피처 후보를 발굴하고, 통계적·행동적 근거를 통해 검증합니다.

### EDA Notebooks

- `EDA_CARD.ipynb`  
  카드 속성 기반 리스크 구조 분석

- `EDA_ERR.ipynb`  
  오류(errors) 패턴 기반 리스크 분석  
  (예: CVV 오류, PIN 오류 등)

- `EDA_HISTORY.ipynb`  
  카드/클라이언트 단위 과거 이력 기반 분석

- `EDA_INTERACTION.ipynb`  
  상호작용 피처(곱, 조건부 결합 등) 구조 분석

- `EDA_MCC.ipynb`  
  MCC 리스크 분석  
  - raw fraud rate  
  - Bayesian smoothing  
  - risk level 정의

- `EDA_MONEY.ipynb`  
  금액/소득/한도 기반 비율 및 로그 스케일 분석

- `EDA_PERSON.ipynb`  
  개인 속성(나이, 성별, 은퇴 등) 기반 분석

- `EDA_TIME.ipynb`  
  시간 기반 리스크 구조  
  - weekday / hour  
  - cyclic encoding  
  - 위험 시간대 정의

### Feature Selection

- `FEATURESELECT.ipynb`  
  모든 EDA 결과를 종합해 최종 피처 셋을 확정  
  (keep / drop 근거 정리 및 검증)

---

## 2) PIPELINE

EDA에서 확정된 로직을 실제 학습·운영 가능한 형태로 구현한 실행 파이프라인 영역입니다.

---

# 2. STAGE1 — Transaction-Only Layer

거래 단독 정보만으로 구성되는 1차 판별 레이어입니다.

## 목적

- 빠른 스코어링
- Stage2로 넘길 Review 대상 선별
- 최소 정보 기반의 안정적 1차 필터

---

## STAGE1 구성

### DATA1.py

거래 데이터 정제 및 기본 파생 피처 생성

주요 처리:

- 온라인 거래 필터링 (`is_online`)
- 라벨 JSON 스트리밍 로드 후 merge
- 금액 정리 및 `log_abs_amount`
- 에러 플래그 생성 (`err_bad_cvv` 등)
- 시간 파생 (`tx_year`, `tx_month`, `tx_hour`, `weekday`)
- MCC 정리

---

### DATA2.py

Stage1 artifacts 생성 및 Stage1 데이터셋 빌드

Artifacts 핵심 구성:

- `mcc_smoothed_risk_map`
- `mcc_risk_level`
- `risky_wd_hour`
- `high_risk_months`

Stage1 최종 피처 예:

- `log_abs_amount`
- `hour_sin`, `hour_cos`
- `month_sin`
- `mcc_smoothed_risk`
- `mcc_risk_level`
- `is_risky_wd_hour`
- `err_bad_cvv`

---

### MODEL.ipynb

Stage1 최종 모델 확정

- Logistic Regression 기반 최적 설정 선택
- 운영 threshold 확정
- Stage2로 전달할 Review ids 생성

---

# 3. STAGE2 — Context & History Layer

사용자·카드 컨텍스트 및 거래 히스토리를 포함하는 2차 판별 레이어입니다.

## 목적

- Stage1에서 선별된 거래에 대해 정밀 판별
- 행동 패턴 기반 이상 탐지
- 속도·신규성·변화 기반 리스크 강화

---

## STAGE2 구성

### DATA0.py

Stage2 베이스 테이블 생성

거래 + 사용자 + 카드 데이터를 결합하고 다음을 생성:

- 계정/만료 파생  
  - `months_to_expire`
  - `months_from_account`
  - `years_since_pin_change`
  - `years_to_retirement`

- 카드 속성  
  - `is_credit`, `is_prepaid`
  - `has_chip`
  - `cb_Visa`, `cb_Mastercard`, `cb_Amex`, `cb_Discover`

- 금액/소득/한도 비율  
  - `amount_income_ratio`
  - `amount_limit_ratio`
  - 로그 변환

---

### DATA1.py (TRAIN_stage2)

히스토리 기반 피처 생성

핵심 특징:

- 과거 거래 기반 shift/rolling 구조
- 속도 급증 (`velocity_spike_ratio`)
- 신규성 (`client_mcc_is_new`, `merchant_is_new`)
- 변화 (`merchant_change_last5`)
- 간격 기반 이상 (`log_interval_dev`)
- 극단 한도 사용 (`limit_ratio_extreme`)

Stage1 피처를 merge하여 최종 TRAIN_stage2 완성

---

### DATA2.py (TEST_stage2)

Stage1 Review ids만 Stage2로 전달

중요 설계:

- 히스토리 계산은 전체 raw 데이터를 기준으로 수행
- 마지막 단계에서 Review ids만 필터링

---

### MODEL.ipynb

Stage2 최종 모델 확정

- LightGBM 기반 최적 모델 선택
- 최종 threshold 확정

---

# 4. CHECK — Drift Validation

운영 환경에서의 분포 이동 대응을 검증하는 영역입니다.

구성:

- Stage2 데이터 생성 재현
- Drift 지표 계산
- 분포 이동 발생 시 대응 전략 실험

목적:

- 데이터 분포 변화 감지
- artifacts 업데이트 여부 판단
- threshold 재조정 필요성 검증

---

# 5. End-to-End Logic

1. EDA에서 피처 후보 발굴
2. Feature Selection으로 최종 피처 확정
3. Stage1 데이터 생성
4. Stage1 모델 학습 및 Review ids 선별
5. Stage2 데이터 생성 (히스토리 포함)
6. Stage2 모델 학습 및 최종 판별
7. Drift 감지 및 대응 전략 검증

---

# 6. Design Philosophy

- 2-Stage 구조는 **속도와 정밀도의 분리 설계**
- Stage1은 거래 단독 기반의 빠른 필터
- Stage2는 컨텍스트 + 행동 기반의 심층 분석
- 모든 히스토리 피처는 시간 정렬 후 생성
- 운영 안정성을 고려한 artifacts 기반 설계
- Drift 대응을 고려한 구조적 확장 가능성 확보