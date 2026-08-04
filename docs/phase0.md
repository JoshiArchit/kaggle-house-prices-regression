# House Prices: Phase 0

## Problem Statement

Predict residential home sale prices (`SalePrice`) using the Kaggle "House Prices: Advanced Regression Techniques" dataset. The competition metric is RMSE between the log of predicted and actual sale price, so all modeling here targets `log1p(SalePrice)` rather than raw price.

## EDA Findings

**Target distribution.** `SalePrice` is right-skewed: a long tail of expensive homes pulls the distribution away from normal. `log1p(SalePrice)` is visibly more symmetric, which motivated modeling on the log target throughout, consistent with the competition's own metric.

**Outliers.** A scatter of `GrLivArea` vs. `SalePrice` revealed two homes with unusually large living area but low sale price, breaking the otherwise-clear upward trend. Investigating `SaleCondition` for those rows showed `Partial`, meaning the sale was recorded before construction was complete, a structurally different transaction type than a normal completed-home sale. These two rows were removed, since they don't represent the population the model is meant to predict.

A full numeric-feature-vs-`SalePrice` scatter grid was also reviewed. One additional anomaly was found and corrected: a `GarageYrBlt` value of ~2207 (an evident data-entry typo, most plausibly 2007), on the same row already excluded via the `GrLivArea`/`Partial` filter. A separate high-`TotalBsmtSF`/lower-price point was investigated but had `SaleCondition = Normal` with no other distinguishing evidence, so it was kept, since removal without a concrete justification would have been an unsupported judgment call.

**Missing values.** 19 columns had missing data, falling into two distinct categories:

- **Structural absence** (`PoolQC`, `MiscFeature`, `Alley`, `Fence`, `MasVnrType`, `FireplaceQu`, and the Garage/Basement quality-and-type columns): `data_description.txt` confirms `NA` explicitly denotes the feature does not exist (e.g. "NA = No Pool"), not an unrecorded value. Filled with an explicit `"None"` category. Verified for consistency where checkable, e.g. all 81 rows missing `GarageType` also missing `GarageYrBlt`.
- **Genuinely missing** (`LotFrontage`, `GarageYrBlt`, `MasVnrArea`, `Electrical`): every home has these attributes, so a blank means an unrecorded value, not an absent feature. `LotFrontage` was imputed by neighborhood median rather than a single global median, since street frontage correlates with location. `GarageYrBlt` and `MasVnrArea` were filled with `0` (no garage / no veneer). `Electrical`'s single missing row was filled with the column mode.

**Correlation with SalePrice.** `OverallQual` (0.80) and `GrLivArea` (0.73) are the strongest linear predictors, consistent with the scatter-plot trends observed directly. Several size-related features (`TotalBsmtSF`, `GarageCars`, `1stFlrSF`, `GarageArea`) cluster in the 0.6 range, expected since they measure related aspects of home size and are likely to be collinear. `OverallCond` showed a near-zero, slightly negative correlation, a known quirk of this dataset rather than a data issue; `OverallCond` and `OverallQual` capture different things, and this loose relationship was noted rather than "fixed."

## Baseline Results

Target: `log1p(SalePrice)`. Feature set: cleaned numeric columns plus one-hot encoded categoricals (`Neighborhood` and related fields). 80/20 train/validation split.

| Model                          | Validation RMSE (log scale) |
| ------------------------------ | --------------------------- |
| Linear Regression              | 0.1292                      |
| Random Forest (default params) | 0.1489                      |
| XGBoost (default params)       | 0.1518                      |

**Note:** Linear Regression outperforming both tree-based models is a real, explainable result rather than a bug, likely due to the small dataset size (~1460 rows) and a feature set already dominated by features with strong linear relationships to the target. Untuned tree ensembles can overfit small datasets more easily than a low-variance linear model. This is expected to change once hyperparameter tuning is introduced.

## What I'd Improve in Phase 1

- Hyperparameter tuning for Random Forest and XGBoost (e.g. via `GridSearchCV` or `Optuna`), expected to close or reverse the current gap with Linear Regression.
- Fuller feature engineering: only a subset of available categorical columns were used for the baseline; several likely carry useful signal (e.g. `ExterQual`, `KitchenQual`, `BsmtQual` as ordinal-encoded features rather than dropped or one-hot'd flatly).
- Address multicollinearity among the size-related numeric features before relying further on linear models, since several sit at 0.6+ correlation with each other as well as with the target.
- Revisit outlier review per-feature more exhaustively rather than the single visual pass done here, particularly for features that showed isolated high-leverage points without a confirmed explanation.
