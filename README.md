
# 📌 Fraud Risk Reduction Project

**Designing an Explainable Pre-Detection Warning System for Card Fraud**

---

## 📍 Project Overview

**Goal**
거래 데이터 기반으로 **Fraud 위험 신호를 사전에 감지**하고,
**Fraud 유형별로 차별화된 대응이 가능한 Warning System을 설계**한다.

본 프로젝트는 단순한 Fraud 분류 모델이 아니라,
**사전 예방(Pre-Detection)**과 **의사결정 설명 가능성**을 핵심 목표로 둔다.

---

## 🎯 One-Line Definition

> 거래 로그에서 **Fraud 발생 이전의 위험 신호(Indicator)**를 정의하고,
> Rule + Score + Policy 구조의 **Fraud Warning System**을 설계한다.

---

## 0️⃣ Stakeholder Analysis & Terminology

### 0.1 Stakeholders

### ① 카드사 (Project Owner)

**Goal**

* Fraud 사고 감소
* 고객 신뢰도 및 서비스 안정성 확보

**Needs**

* **Fraud Pre-Detection Warning System**

  * 사고 발생 *이전* 위험 신호 탐지
* **Fraud Analysis Report**

  * Fraud 유형별 핵심 signal
  * 공통 패턴과 신규 유형 탐색
  * 모델 결과 해석 자료

> 모델은 통계를 대체하지 않으며,
> **비선형 패턴 탐색 및 신호 보강 도구**로 활용

**Deliverables**

* Fraud Warning Model
* Fraud 유형 기반 분석 리포트
* 유형별 대응 전략 제안

---

### ② 고객 (End User)

**Goal**

* 거래 안전성 보장

**Needs**

* 안정적인 예측 성능
* 다양한 거래 환경에서도 일관된 동작
* 과도한 오탐(False Positive) 최소화

---

### 0.2 Core Terminology

**Detection Model**

* 거래가 Fraud인지 여부를 *사후적으로* 분류
* 목적: 패턴 탐색, feature 중요도 분석

**Warning System**

* Fraud 발생 *이전* 위험 신호 감지
* 목적: 사고 예방 및 사전 개입

**Indicator**

* Fraud 위험을 설명·정량화하는 신호
* 모델 없이도 해석 가능한 Rule 기반 요소 포함

---

## 1️⃣ Main Objective

Card Fraud를 **사전에 감지**하고,
**Fraud 유형별 대응이 가능한 Warning System 구축**

---

## 2️⃣ Phase 0 – Hypothesis Setting

Fraud는 무작위 사건이 아니라
**반복되는 구조적 패턴을 가진 사건**이라는 가설에서 출발

### Example Hypotheses

* 짧은 시간 내 장거리 결제 발생 → Fraud 확률 증가
* 갑작스러운 고액 결제 vs 반복적 소액 결제 → Fraud 유형 차이
* Error 발생 이후 정상 거래 대비 Fraud 비율 상승

> 모든 가설은 **EDA 및 모델 분석을 통해 검증 또는 기각**

---

## 3️⃣ Phase 1 – Indicator Discovery (Core Phase)

### 사고 역순 접근

Fraud 감소
→ 사전 위험 감지 필요
→ **Fraud Warning Indicator 정의**

---

### 3.1 Post-Mortem EDA

**Objective**

* Fraud(1)의 구조적 특징 파악

**Methods**

* Fraud / Non-Fraud 분포 비교
* Logistic Regression 계수, 신뢰구간, p-value 분석
* Fraud 발생 전·후 거래 패턴 변화 분석
* 가설 검증

**Outputs**

* Fraud 민감 feature 후보군
* 파생 변수 설계 근거

---

### 3.2 Detection Model (Analysis Tool)

> ⚠️ Warning 모델이 아닌 **분석용 모델**

**Design Principles**

* 극단적 불균형 대응

  * Fraud 전량 사용
  * Non-Fraud 언더샘플링
* 다중 랜덤 샘플링 데이터셋 구성
* 데이터셋별 독립 모델 학습

**Analysis**

* SHAP / Feature Importance
* 데이터셋 간 공통 중요 feature Voting

**Outputs**

* 선형/비선형 관점에서 일관된 Fraud signal
* Indicator 후보 확정

---

### 3.3 Phase 1 Output

* Fraud 발생 조건 요약
* 핵심 Fraud Warning Indicator 정의
* 가설 검증 결과 정리

---

## 4️⃣ Phase 2 – Fraud Warning System Design

### Problem Statement

* 단일 Black-box 예측은 실무 니즈에 부적합
* **Fraud 유형별 대응 전략 필요**

---

### 4.1 Fraud Clustering

* Fraud 거래만 대상으로 Clustering
* 목적: Fraud 유형 분리
* 방법: k-means 등 (선정 기준 명시)

---

### 4.2 Cluster-Based Analysis

* Cluster별 주요 feature 분석
* Fraud 유형별 EDA
* 유형별 핵심 Indicator 재정의
* 필요 시 Multi-head Model 활용

**Outputs**

* Fraud 유형 정의
* 유형별 위험 Indicator 세분화

---

## Detection Model vs Warning Model

| 구분    | Detection Model | Warning Model              |
| ----- | --------------- | -------------------------- |
| 목적    | 사후 분류           | **사전 예방**                  |
| 타이밍   | After           | **Before**                 |
| 형태    | ML 중심           | **Rule + Score + Context** |
| 설명성   | 낮아도 가능          | **필수**                     |
| 오탐 허용 | 비교적 가능          | **매우 민감**                  |

> Warning 모델의 핵심은 **정확도보다 의사결정 구조**

---

### Warning System Architecture

```
[Indicator Layer]
      ↓
[Risk Scoring Layer]
      ↓
[Decision Policy Layer]
      ↓
[Action: Approve / Monitor / Alert / Block]
```

**Indicator Layer**

* “값이 증가할수록 Fraud 위험이 증가한다”는 명확한 정의

**Risk Scoring Layer**

* Rule-weighted Score
* Lightweight Model (Logistic / GBM / GAM)

**Decision Policy Layer**

| Risk Score | Action            |
| ---------- | ----------------- |
| < T1       | 정상 승인             |
| T1 – T2    | Silent Monitoring |
| T2 – T3    | 고객 알림             |
| ≥ T3       | 거래 차단             |

**Action Layer**

* 모든 조치에 대해 설명 가능해야 함

---

## 5️⃣ Phase 3 – Simulation Test

**Purpose**

* 시스템 안정성 및 정책 타당성 검증

**Methods**

* Time-based Replay
* Policy Sweep (Threshold / Alert Budget)
* Segment Stress Test

**Key Metrics**

* Recall@K (Alert Budget 기반)
* Cost-based FP vs FN Trade-off
* Lead Time (사전성 지표)

---

## 6️⃣ Final Deliverables

* Fraud Warning Model
* Fraud 유형별 대응 전략 Report
* Visualization Dashboard (Tableau)

---

## 🗂 How to Deal with Parquet

대용량 거래 데이터를 효율적으로 다루기 위한 기본 전처리 가이드

```python
import pandas as pd

df = pd.read_parquet("transactions_clean.parquet")

# Datetime
df["date"] = pd.to_datetime(df["date"])

# ID columns
df["client_id"]   = df["client_id"].astype("int32")
df["card_id"]     = df["card_id"].astype("int32")
df["merchant_id"] = df["merchant_id"].astype("int32")
df["mcc"]         = df["mcc"].astype("int16")

# Amount
df["amount"] = df["amount"].astype("float32")

# Categorical features
for c in ["use_chip", "merchant_city", "merchant_state", "zip"]:
    df[c] = df[c].astype("category")

# Error flags
for c in [
    "has_error",
    "err_card_credential",
    "err_authentication",
    "err_financial",
    "err_system"
]:
    df[c] = df[c].astype("int8")

# Target
df["fraud"] = df["fraud"].astype("int8")

df.info(memory_usage="deep")
```

**Purpose**

* 메모리 사용량 최소화
* 대규모 EDA 및 시뮬레이션 환경 안정성 확보

---