# 🏡 미국 주택 가격 예측 프로젝트

## 📌 프로젝트 개요

이 프로젝트는 미국 주택 가격을 예측하는 머신러닝 모델을 구축하는 것입니다.

- **데이터셋**: Kaggle의 `House Prices - Advanced Regression Techniques`
- **목표 변수**: `SalePrice` (로그 변환된 값 사용)
- **사용한 기법**: 데이터 전처리, Feature Engineering, 하이퍼파라미터 튜닝, 모델 평가 및 교차검증
- **최종 예측 모델**: `XGBoost Regressor`

## 📂 데이터 설명

- `train.csv`: 학습 데이터 (1460개 샘플, 80개 특성)
- `test.csv`: 예측을 위한 테스트 데이터 (1459개 샘플, 80개 특성)
- `sample_submission.csv`: 예측 결과 제출 형식

## ⚙️ 프로젝트 실행 방법

### 1️⃣ **필요한 라이브러리 설치**

프로젝트 실행을 위해 아래 패키지가 필요합니다.

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

### 2️⃣ **Python 스크립트 실행**

```bash
python house_price_prediction.py
```

또는 Jupyter Notebook에서 실행 가능

---

## 📊 데이터 전처리

1. **결측치 처리**

   - 범주형 변수 → `'None'`으로 대체 (`GarageType`, `BsmtQual` 등)
   - 수치형 변수 → 중앙값 대체 (`LotFrontage`, `GarageYrBlt`, `MasVnrArea` 등)
   - `GarageYrBlt`가 없는 경우 0으로 설정
   - 최빈값 대체 (`Electrical` 변수)
2. **Feature Engineering**

   - 로그 변환 (`SalePrice`, 왜도가 큰 변수)
   - 새로운 변수 생성 (`TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`)
   - 다중공선성(VIF)이 높은 변수 제거
3. **데이터 인코딩 및 표준화**

   - 범주형 변수 → `OneHotEncoding`
   - 수치형 변수 → `StandardScaler` 적용

---

## 🏗️ 모델 구축 및 평가

| 모델                        | RMSE    |
| --------------------------- | ------- |
| **Linear Regression** | `XXX` |
| **Random Forest**     | `XXX` |
| **XGBoost**           | `XXX` |

### 1️⃣ 모델별 평가 지표

- **RMSE (Root Mean Squared Error) 사용**
- **K-Fold Cross Validation 적용** (5-Fold)

### 2️⃣ 최적 모델 (`XGBoost`)

```python
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train_preprocessed, y_train)
```

---

## 📤 Kaggle 제출 파일 생성

```python
def create_submission(model, X_test, test_df, model_name):
    """ 테스트 데이터에 대한 예측을 수행하고 Kaggle 제출 파일 생성 """
    y_pred = model.predict(X_test)
    y_pred = np.expm1(y_pred)
    submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": y_pred})
    file_name = f"submission_{model_name}.csv"
    submission.to_csv(file_name, index=False)
    print(f"✅ {model_name} 제출 파일 저장 완료: {file_name}")
```

최적 모델을 사용하여 최종 예측값을 제출 파일로 저장합니다.

---

## 📌 결론

- `XGBoost`가 가장 낮은 RMSE를 기록하여 최적 모델로 선정됨
- Feature Engineering을 통해 일부 변수 추가 (`TotalSF` 등)
- 최적의 하이퍼파라미터를 적용하여 모델 성능 개선

✅ **추가 개선 가능성**: 모델 앙상블(Stacking, Blending) 또는 더 정교한 Feature Engineering 활용 가능

🚀 **Kaggle 제출 후 결과 확인 필수!**

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submissions#

![주택 가격 예측](images/screenshot.png)
