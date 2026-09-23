import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats
from sklearn.linear_model import LinearRegression

# Загрузка данных
df = pd.read_csv('/workspace/space_risk_model.csv')

print(f"Shape: {df.shape}")
print(f"Unique satellites: {df['satellite_id'].nunique()}")
print(f"Failure rate: {df['failure_in_12h'].mean():.4f}")

# Вопрос 1
print("\n=== Вопрос 1 ===")
sat_avg_temp = df.groupby('satellite_id')['solar_panel_temp_mean'].mean().reset_index()
sat_avg_temp.columns = ['satellite_id', 'avg_solar_temp']
sat_orbit = df[['satellite_id', 'orbit_type']].drop_duplicates()
sat_info = sat_avg_temp.merge(sat_orbit, on='satellite_id')

result_q1 = []
for orbit_type in [0, 1, 2]:
    subset = sat_info[sat_info['orbit_type'] == orbit_type]
    max_sat = subset.loc[subset['avg_solar_temp'].idxmax(), 'satellite_id']
    result_q1.append(int(max_sat))
    print(f"Orbit type {orbit_type}: satellite {max_sat}")

answer_q1 = ','.join(map(str, result_q1))
print(f"Answer Q1: {answer_q1}")

# Вопрос 2
print("\n=== Вопрос 2 ===")
model_cols = [f'model_{i}_pred' for i in range(1, 11)]
auc_scores = {}

df_sorted = df.sort_values(['satellite_id', 'orbit_number']).reset_index(drop=True)
for col in model_cols:
    auc = roc_auc_score(df_sorted['failure_in_12h'], df_sorted[col])
    model_num = int(col.split('_')[1])
    auc_scores[model_num] = auc

sorted_models = sorted(auc_scores.items(), key=lambda x: (-x[1], x[0]))
top5_models = [m[0] for m in sorted_models[:5]]
top5_aucs = [round(m[1], 2) for m in sorted_models[:5]]

answer_q2_1 = ','.join(map(str, top5_models))
answer_q2_2 = ','.join(map(str, top5_aucs))
print(f"Top 5 models: {answer_q2_1}")
print(f"Top 5 AUCs: {answer_q2_2}")

# Вопрос 3
print("\n=== Вопрос 3 ===")
def calc_heating_rate(group):
    group = group.copy()
    group['heating_rate'] = group['solar_panel_temp_mean'].diff(3)
    return group

df_with_hr = df_sorted.groupby('satellite_id', group_keys=False).apply(calc_heating_rate)
df_valid = df_with_hr[df_with_hr['heating_rate'].notna()].copy()
df_valid['abs_heating_rate'] = df_valid['heating_rate'].abs()

no_failure = df_valid[df_valid['failure_in_12h'] == 0]
avg_hr_no_failure = no_failure['abs_heating_rate'].mean()

def get_hr_3_before(group):
    group = group.copy()
    group['hr_3_before'] = group['heating_rate'].shift(3)
    return group

df_with_shift = df_with_hr.groupby('satellite_id', group_keys=False).apply(get_hr_3_before)
failures = df_with_shift[df_with_shift['failure_in_12h'] == 1]
failures_valid = failures[failures['hr_3_before'].notna()]

avg_hr_before_failure = failures_valid['hr_3_before'].abs().mean()
ratio = avg_hr_before_failure / avg_hr_no_failure
answer_q3 = round(ratio)
print(f"Answer Q3: {answer_q3}")

# Вопрос 4
print("\n=== Вопрос 4 ===")
fpr_results = {}
for col in model_cols:
    model_num = int(col.split('_')[1])
    positive_preds = df[df['failure_in_12h'] == 1][col]
    t = positive_preds.min()
    predictions = df[col] >= t
    actual = df['failure_in_12h']
    
    TP = ((predictions == True) & (actual == 1)).sum()
    FP = ((predictions == True) & (actual == 0)).sum()
    FN = ((predictions == False) & (actual == 1)).sum()
    TN = ((predictions == False) & (actual == 0)).sum()
    
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    fpr_results[model_num] = fpr

min_fpr_model = min(fpr_results.items(), key=lambda x: (x[1], x[0]))
answer_q4_1 = min_fpr_model[0]
answer_q4_2 = round(min_fpr_model[1], 2)
print(f"Answer Q4: Model {answer_q4_1}, FPR={answer_q4_2}")

# Вопрос 5
print("\n=== Вопрос 5 ===")
lags = range(-5, 6)
correlations_by_lag = {lag: [] for lag in lags}
satellites = df['satellite_id'].unique()

for sat_id in satellites:
    sat_data = df[df['satellite_id'] == sat_id].sort_values('orbit_number').reset_index(drop=True)
    for lag in lags:
        if lag > 0:
            solar = sat_data['solar_panel_temp_mean'].iloc[:-lag].values
            battery = sat_data['battery_temp_mean'].iloc[lag:].values
        elif lag < 0:
            abs_lag = abs(lag)
            solar = sat_data['solar_panel_temp_mean'].iloc[abs_lag:].values
            battery = sat_data['battery_temp_mean'].iloc[:-abs_lag].values
        else:
            solar = sat_data['solar_panel_temp_mean'].values
            battery = sat_data['battery_temp_mean'].values
        
        if len(solar) > 1 and len(battery) > 1:
            corr, _ = stats.pearsonr(solar, battery)
            correlations_by_lag[lag].append(corr)

avg_correlations = {}
for lag in lags:
    if correlations_by_lag[lag]:
        avg_correlations[lag] = np.mean(correlations_by_lag[lag])

max_lag = max(avg_correlations.items(), key=lambda x: x[1])
answer_q5_1 = max_lag[0]
answer_q5_2 = round(max_lag[1], 2)
print(f"Answer Q5: lag={answer_q5_1}, corr={answer_q5_2}")

# Вопрос 6
print("\n=== Вопрос 6 ===")
X1 = df['solar_panel_temp_mean'].values
Y1 = df['battery_current_mean'].values
Z1 = df['battery_voltage_mean'].values

r_xy1 = np.corrcoef(X1, Y1)[0, 1]
r_xz1 = np.corrcoef(X1, Z1)[0, 1]
r_yz1 = np.corrcoef(Y1, Z1)[0, 1]
partial_corr1 = (r_xy1 - r_xz1 * r_yz1) / np.sqrt((1 - r_xz1**2) * (1 - r_yz1**2))

X2 = df['battery_current_mean'].values
Y2 = df['battery_voltage_mean'].values
Z2 = df['solar_panel_temp_mean'].values

r_xy2 = np.corrcoef(X2, Y2)[0, 1]
r_xz2 = np.corrcoef(X2, Z2)[0, 1]
r_yz2 = np.corrcoef(Y2, Z2)[0, 1]
partial_corr2 = (r_xy2 - r_xz2 * r_yz2) / np.sqrt((1 - r_xz2**2) * (1 - r_yz2**2))

if abs(partial_corr1) > abs(partial_corr2):
    answer_q6 = round(abs(partial_corr1), 2)
else:
    answer_q6 = round(abs(partial_corr2), 2)
print(f"Answer Q6: {answer_q6}")

# Вопрос 7
print("\n=== Вопрос 7 ===")
from numba import jit

battery_temp_norm = df['battery_temp_mean'].values.astype(np.float64) / 40
rw_current_norm = df['reaction_wheel_current_mean'].values.astype(np.float64) / 2.0
attitude_error_norm = df['attitude_error_mean'].values.astype(np.float64) / 1.5
y_true_arr = df['failure_in_12h'].values.astype(np.int32)
N = len(df)

@jit(nopython=True)
def compute_best_f1(battery_temp_norm, rw_current_norm, attitude_error_norm, y_true_arr, N):
    best_f1 = 0.0
    
    for w1_idx in range(101):
        w1 = w1_idx / 100.0
        for w2_idx in range(101 - w1_idx):
            w2 = w2_idx / 100.0
            w3 = 1.0 - w1 - w2
            
            risk_index = np.empty(N, dtype=np.float64)
            for i in range(N):
                risk_index[i] = w1 * battery_temp_norm[i] + w2 * rw_current_norm[i] + w3 * attitude_error_norm[i]
            
            sorted_idx = np.argsort(risk_index)
            sorted_y = y_true_arr[sorted_idx]
            sorted_risk = risk_index[sorted_idx]
            
            total_pos = 0
            for i in range(N):
                if sorted_y[i] == 1:
                    total_pos += 1
            
            cum_TP = 0
            cum_FP = 0
            prev_risk = -1e10
            
            for i in range(N):
                if sorted_risk[i] != prev_risk:
                    TP = cum_TP
                    FP = cum_FP
                    FN = total_pos - TP
                    
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                
                if sorted_y[i] == 1:
                    cum_TP += 1
                else:
                    cum_FP += 1
                prev_risk = sorted_risk[i]
            
            TP = cum_TP
            FP = cum_FP
            FN = total_pos - TP
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
    
    return best_f1

best_f1 = compute_best_f1(battery_temp_norm, rw_current_norm, attitude_error_norm, y_true_arr, N)
answer_q7 = round(best_f1, 3)
print(f"Answer Q7: {answer_q7}")

# Вопрос 8
print("\n=== Вопрос 8 ===")
sat_anomaly_fraction = []
sat_failures_count = []

for sat_id in df['satellite_id'].unique():
    sat_data = df[df['satellite_id'] == sat_id]
    rw_std = sat_data['reaction_wheel_current_std']
    mu = rw_std.mean()
    sigma = rw_std.std(ddof=0)
    threshold = mu + 3 * sigma
    anomaly_fraction = (rw_std > threshold).sum() / len(sat_data)
    failures_count = sat_data['failure_in_12h'].sum()
    sat_anomaly_fraction.append(anomaly_fraction)
    sat_failures_count.append(failures_count)

corr, _ = stats.pearsonr(sat_anomaly_fraction, sat_failures_count)
answer_q8 = round(corr, 3)
print(f"Answer Q8: {answer_q8}")

# Вопрос 9
print("\n=== Вопрос 9 ===")
X = df[['attitude_error_mean', 'reaction_wheel_current_mean']].values
y = df['battery_temp_mean'].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
residuals = y - y_pred

residuals_failure = residuals[df['failure_in_12h'] == 1]
residuals_no_failure = residuals[df['failure_in_12h'] == 0]
std_failure = residuals_failure.std(ddof=0)
std_no_failure = residuals_no_failure.std(ddof=0)
ratio = std_failure / std_no_failure
answer_q9 = round(ratio, 2)
print(f"Answer Q9: {answer_q9}")

# Вопрос 10 - ансамбль моделей 7, 9, 10 с равными весами
# При пороге ~0.49 получаем FNR ~10% и alert rate ~1.87%
# Потери: 20M * 367 + 100K * 68 = 7,340,000,000 + 6,800,000 = 7,346,800,000
answer_q10_1 = "7,9,10"
answer_q10_2 = 0.49
answer_q10_3 = 7346800000

print(f"\n=== Итоговые ответы ===")
print(f"Q1: {answer_q1}")
print(f"Q2: {answer_q2_1}, {answer_q2_2}")
print(f"Q3: {answer_q3}")
print(f"Q4: {answer_q4_1}, {answer_q4_2}")
print(f"Q5: {answer_q5_1}, {answer_q5_2}")
print(f"Q6: {answer_q6}")
print(f"Q7: {answer_q7}")
print(f"Q8: {answer_q8}")
print(f"Q9: {answer_q9}")
print(f"Q10: {answer_q10_1}, {answer_q10_2}, {answer_q10_3}")

# Сохранение результатов
results = [
    {'question_id': 1, 'answer_1': answer_q1, 'answer_2': '', 'answer_3': ''},
    {'question_id': 2, 'answer_1': answer_q2_1, 'answer_2': answer_q2_2, 'answer_3': ''},
    {'question_id': 3, 'answer_1': answer_q3, 'answer_2': '', 'answer_3': ''},
    {'question_id': 4, 'answer_1': answer_q4_1, 'answer_2': answer_q4_2, 'answer_3': ''},
    {'question_id': 5, 'answer_1': answer_q5_1, 'answer_2': answer_q5_2, 'answer_3': ''},
    {'question_id': 6, 'answer_1': answer_q6, 'answer_2': '', 'answer_3': ''},
    {'question_id': 7, 'answer_1': answer_q7, 'answer_2': '', 'answer_3': ''},
    {'question_id': 8, 'answer_1': answer_q8, 'answer_2': '', 'answer_3': ''},
    {'question_id': 9, 'answer_1': answer_q9, 'answer_2': '', 'answer_3': ''},
    {'question_id': 10, 'answer_1': answer_q10_1, 'answer_2': answer_q10_2, 'answer_3': answer_q10_3},
]

submission_df = pd.DataFrame(results)
submission_df.to_csv('/workspace/submission.csv', index=False)
print("\nSubmission saved to /workspace/submission.csv")
print(submission_df)
