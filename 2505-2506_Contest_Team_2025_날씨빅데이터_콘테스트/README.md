# 지역난방 열수요 예측: 고도화된 스태킹 앙상블

19개 지사의 시간별 지역난방 열수요를 예측하는 시즌별 분할 스태킹 앙상블 모델입니다. 난방철과 비난방철의 수요 패턴 차이를 고려한 특화 모델링으로 높은 예측 정확도를 달성합니다.

## < 프로젝트 개요 >

- **데이터**: 2021-2023년 훈련 데이터 (499,301개), 2024년 테스트 데이터 (166,915개)
- **목표**: 19개 지사별 시간별 열수요 예측 (RMSE 최소화)
- **모델링 전략**: 시즌별 분할 + 스태킹 앙상블 + Huber Loss 최적화
- **최종 성능**: 난방철 RMSE 22.63, 비난방철 RMSE 9.76

## < 모델링 설계 >

### 1. 시즌별 분할 모델링 전략

### 배경 및 필요성

지역난방 수요는 계절에 따라 극명한 차이를 보여 단일 모델로는 정확한 예측이 어렵습니다.

**데이터 분석 결과:**

- **난방철 (10-4월)**: 평균 135.94, 표준편차 135.29 (289,997개)
- **비난방철 (5-9월)**: 평균 40.35, 표준편차 32.00 (209,304개)
- **수요 비율**: 난방철이 비난방철 대비 **3.4배** 높음

### 기술적 근거

1. **데이터 불균형**: 높은 난방 수요에 편향되어 비난방철 예측 정확도 저하
2. **패턴 차이**: 난방철은 온도 의존적 비선형, 비난방철은 안정적 베이스라인
3. **Feature 중요도**: 계절별로 주요 예측 변수의 영향력이 상이함

### 2. Huber Loss 기반 최적화

### 선택 배경

평가 지표는 RMSE이지만, 훈련에는 **Huber Loss (δ=1.0)**를 사용했습니다.

### 객관적 근거

1. **로버스트 회귀 이론** (Huber, 1964):
    - 작은 오차: L2 손실 (제곱 오차)
    - 큰 오차: L1 손실 (절댓값) → 이상치에 덜 민감
2. **에너지 수요 예측 연구** (Zhang et al., 2019):
    - 건물 에너지 수요 예측에서 Huber Loss가 RMSE 성능을 5-12% 개선
    - 극값이 존재하는 시계열 데이터에서 특히 효과적
3. **실증 검증**:
    
    `Huber Loss 훈련 vs MSE 훈련:
    - 난방철: 22.63 vs 24.1 (약 6% 개선)
    - 비난방철: 9.76 vs 10.8 (약 10% 개선)`
    

## < 모델 아키텍처 >

### 스태킹 앙상블 구조

`시즌별 분할
├── 난방철 모델 (10월~4월)
│   ├── Level 0: Prophet + CatBoost
│   └── Level 1: Ridge Meta-learner
└── 비난방철 모델 (5월~9월)
    ├── Level 0: Prophet + CatBoost  
    └── Level 1: Ridge Meta-learner`

### 모델 선택 근거

### CatBoost (Tree-based) 선택 이유

1. **비선형 관계 포착**: 온도-수요 관계의 구간별 다른 기울기
2. **범주형 변수 처리**: 지사별, 시간대별 특성 자동 처리
3. **Feature 상호작용**: 온도×시간, 온도×지사 등 복잡한 교호작용 자동 탐지
4. **연구 근거**: Fernández-Delgado et al. (2014) - Tree 모델이 feature 수와 관계없이 일관된 성능 우위

### Stacking 앙상블 선택 근거

1. **이질적 모델 조합**: Prophet(시계열 구조) + CatBoost(복잡한 패턴)의 상호 보완
2. **동적 가중치**: Meta-learner가 각 모델의 강점을 상황별로 조합
3. **연구 근거**: Wolpert (1992) - Stacked Generalization이 단순 평균보다 우수

## 🔧 고급 기술 구현

### 1. SVR 기반 결측치 보간

### 선택 근거

- **전통적 방법 한계**: 선형 보간(비선형 무시), 평균 대체(분산 손실)
- **SVR 장점**: RBF 커널로 비선형 관계 포착, 시간적 연속성과 계절성 동시 고려
- **연구 근거**: Jönsson & Eklundh (2004) - SVR이 시계열 보간에서 우수한 성능

### 구현 방법

```sql
# 지사별 개별 SVR 모델
for branch in df['branch_id'].unique():
    # 시간 특성: hour, day_of_year, month, dayofweek
    svr = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
    svr.fit(X_train_scaled, y_train_scaled)
```

### 2. 이상치 플래그 변수 설계

### 설계 근거

- **극한 기상 영향**: 한파, 강풍, 폭우 시 수요 패턴 급변
- **물리적 근거**: 극한 추위→수요 급증, 강풍→체감온도 하락, 폭우→습도 영향
- **연구 근거**: Hong & Fan (2016) - 극값 기상 정보가 에너지 수요 예측 정확도 10-15% 향상

### 구현 방법

```sql
# 시즌별, 지사별 임계값 계산
thresholds = {
    'ta_q10': season_data['ta'].quantile(0.10),    # 극한 추위
    'ws_q90': season_data['ws'].quantile(0.90),    # 강풍  
    'rn_day_q90': season_data['rn_day'].quantile(0.90)  # 폭우
}
```

### 3. 고도화된 특성 생성

### 시즌별 특화 변수

**난방철 특성:**

- 체감온도 (Wind Chill): `13.12 + 0.6215*ta - 11.37*(ws*3.6)^0.16 + ...`
- 난방도일 (HDD18): `max(0, 18 - ta)`
- 한파 경보 수준: `ColdAdvisory (-12℃), ColdWarning (-15℃)`
- 난방철 월순환 인코딩

**비난방철 특성:**

- 비난방철 월순환 인코딩
- 일사량 기반 변수
- 습도 관련 불쾌지수

### 시계열 파생 변수

```sql
# Lag 변수
ta_lag_3h, ta_lag_6h, ta_lag_24h

# 이동평균
ta_ma_6h, ta_ma_12h, ta_ma_24h  

# 차분변수
ta_diff_3h, ta_diff_6h

# 일별 통계
daily_ta_min, daily_ta_max, daily_ta_mean, daily_temp_range
```

## < 성능 결과 >

### 최종 성능 (Stacking Ensemble)

| 시즌 | 데이터 수 | RMSE | Huber Loss | MAE | 훈련시간 |
| --- | --- | --- | --- | --- | --- |
| 난방철 | 289,997개 | **22.63** | **13.73** | 14.21 | 8.5분 |
| 비난방철 | 209,304개 | **9.76** | **6.06** | 6.54 | 8.0분 |

### 개별 모델 성능 비교

| 모델 | 난방철 RMSE | 비난방철 RMSE | 평균 RMSE |
| --- | --- | --- | --- |
| Prophet | 29.55 | 11.99 | 20.77 |
| CatBoost | 23.37 | 10.03 | 16.70 |
| **Stacking** | **22.63** | **9.76** | **16.20** |

### 스태킹 개선 효과

- **난방철**: +1.57% 개선 (CatBoost 23.37 → Stacking 22.63)
- **비난방철**: +1.46% 개선 (CatBoost 10.03 → Stacking 9.76)

### Meta-model 가중치

- **CatBoost**: 87.2% (주요 예측 엔진)
- **Prophet**: 15.0% (트렌드/계절성 보완)

## < 사용 방법 >

### 1. 환경 설정

```python
pip install pandas numpy scikit-learn catboost prophet optuna torch
pip install holidays pmdarima statsmodels joblib tqdm seaborn matplotlib
```

### 2. 빠른 시작 (최적 파라미터 사용)

```python
# 1. 데이터 로드 및 전처리
train_groups, test_groups, weather_thresholds = load_processed_data()

# 2. 모델 훈련 (최적 파라미터 적용)
def train_all_groups():
    ensemble_models['heating'] = AdvancedStackingEnsemble(
        season_type="heating", group_name="난방시즌"
    )
    ensemble_models['heating'].fit(
        train_groups['heating'], cv_splits, 
        use_predefined_params=True  # 최적 파라미터 사용
    )

# 3. 예측 실행
pred, individual_pred = ensemble_models['heating'].predict(test_groups['heating'])
```

### 3. 전체 파이프라인 실행

```python
# 데이터 전처리부터 예측까지 전체 실행
train_all_groups()  # 약 16분 소요
```

## 📁 프로젝트 구조

```python
├── 0624_v3_advanced_stacking_ensemble_complete.ipynb  # 메인 노트북
├── saved_models/                    # 훈련된 모델
│   ├── ensemble_heating.pkl         # 난방철 앙상블
│   ├── ensemble_non_heating.pkl     # 비난방철 앙상블  
│   └── full_model_package.pkl       # 전체 패키지
├── dataset/                         # 전처리된 데이터
│   ├── train_heating.csv
│   ├── test_heating.csv
│   └── weather_thresholds.pickle
└── outputs/                         # 결과 파일
    ├── original_structure_submission.csv      # 제출용 (166,915행)
    └── original_structure_predictions.csv     # 상세 결과
```

## < 주요 특징 분석 >

### Feature 중요도 (CatBoost 기준)

**난방철:**

1. branch_id: 51.5% (지역별 특성)
2. daily_ta_max: 14.7% (일최고기온)
3. heating_month_cos: 10.0% (계절성)
4. hour_cos: 7.0% (일일 패턴)

**비난방철:**

1. branch_id: 23.2% (지역별 특성)
2. ta_diff_6h: 19.7% (온도 변화율)
3. day: 21.3% (일별 트렌드)
4. ta_lag_3h: 10.4% (단기 이력)

### 시간대별 수요 패턴

- **피크 시간**: 17시 (34.8), 5시 (34.7)
- **저수요 시간**: 2-4시, 14-16시
- **지사별 편차**: 지역 특성에 따른 패턴 차이

### O~S 지사 복구 결과

```python
지사별 최종 예측 성능:
- 지사 O: 평균 67.74 (난방 97.7, 비난방 25.9)
- 지사 P: 평균 96.11 (난방 124.0, 비난방 57.2)  
- 지사 Q: 평균 54.50 (난방 78.1, 비난방 21.6)
- 지사 R: 평균 14.37 (난방 18.3, 비난방 8.9)
- 지사 S: 평균 12.54 (난방 15.8, 비난방 8.1)
```

## < 최적화된 하이퍼파라미터 >

### 난방철 최적 파라미터

```python
prophet_heating = {
    'changepoint_prior_scale': 0.0026,
    'seasonality_prior_scale': 0.1307,
    'holidays_prior_scale': 5.3995,
    'seasonality_mode': 'multiplicative'
}

catboost_heating = {
    'iterations': 1678,
    'depth': 5, 
    'learning_rate': 0.0575,
    'l2_leaf_reg': 12.26,
    'border_count': 42
}

ridge_heating = {'alpha': 63.51}
```

### 비난방철 최적 파라미터

```python
prophet_non_heating = {
    'changepoint_prior_scale': 0.0011,
    'seasonality_prior_scale': 3.6520,
    'holidays_prior_scale': 0.4544,
    'seasonality_mode': 'additive'
}

catboost_non_heating = {
    'iterations': 1818,
    'depth': 8,
    'learning_rate': 0.0853,
    'l2_leaf_reg': 14.96,
    'border_count': 35
}

```

## < 요구사항 >

- **Python**: 3.8+
- **메모리**: 8GB+ 권장
- **저장공간**: 2GB+ (모델 + 데이터)
- **실행시간**: 전체 파이프라인 ~16분

## < 문의 >

프로젝트 관련 문의사항이 있으시면 이슈를 생성해 주세요.