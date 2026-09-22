
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LinearRegression
from itertools import combinations

# Загрузка данных
df = pd.read_csv('space_risk_model.csv')
print(f"Loaded {len(df)} rows")

# ============================================
# Q1: Спутники с максимальной средней температурой солнечных панелей по типам орбит
# ============================================
sat_avg_temp = df.groupby(['satellite_id', 'orbit_type'])['solar_panel_temp_mean'].mean().reset_index()

q1_result = []
for ot in [0, 1, 2]:
    subset = sat_avg_temp[sat_avg_temp['orbit_type'] == ot]
    max_row = subset.loc[subset['solar_panel_temp_mean'].idxmax()]
    q1_result.append(int(max_row['satellite_id']))

q1_answer = f"{q1_result[0]},{q1_result[1]},{q1_result[2]}"
print(f"Q1: {q1_answer}")

# ============================================
# Q2: Пять лучших моделей по ROC-AUC
# ============================================
model_cols = ['model_1_pred', 'model_2_pred', 'model_3_pred', 'model_4_pred', 'model_5_pred',
              'model_6_pred', 'model_7_pred', 'model_8_pred', 'model_9_pred', 'model_10_pred']

y_true = df['failure_in_12h'].values

auc_scores = {}
for i, col in enumerate(model_cols, 1):
    auc = roc_auc_score(y_true, df[col].values)
    auc_scores[i] = auc

sorted_models = sorted(auc_scores.items(), key=lambda x: (-x[1], x[0]))
top5 = sorted_models[:5]
top5_nums = [m for m, _ in top5]
top5_aucs = [f"{auc:.2f}" for _, auc in top5]

q2_answer1 = ','.join(map(str, top5_nums))
q2_answer2 = ','.join(top5_aucs)
print(f"Q2: models={q2_answer1}, AUCs={q2_answer2}")

# ============================================
# Q3: Отношение средней абсолютной скорости нагрева
# ============================================
df_sorted = df.sort_values(['satellite_id', 'orbit_number']).copy()
df_sorted['heating_rate'] = df_sorted.groupby('satellite_id')['solar_panel_temp_mean'].diff(3)
df_sorted['abs_heating_rate'] = df_sorted['heating_rate'].abs()

no_failure = df_sorted[df_sorted['failure_in_12h'] == 0]['abs_heating_rate'].dropna()
avg_no_failure = no_failure.mean()

failure_rows = df_sorted[df_sorted['failure_in_12h'] == 1].copy()
heating_rates_before_failure = []
for idx, row in failure_rows.iterrows():
    sat_id = row['satellite_id']
    orbit_num = row['orbit_number']
    prev_orbit = orbit_num - 3
    if prev_orbit < 0:
        continue
    mask = (df_sorted['satellite_id'] == sat_id) & (df_sorted['orbit_number'] == prev_orbit)
    matching = df_sorted[mask]
    if len(matching) > 0:
        hr = matching['heating_rate'].values[0]
        if not np.isnan(hr):
            heating_rates_before_failure.append(abs(hr))

avg_before_failure = np.mean(heating_rates_before_failure) if heating_rates_before_failure else 0
q3_answer = round(avg_before_failure / avg_no_failure)
print(f"Q3: {q3_answer}")

# ============================================
# Q4: Модель с минимальным FPR при Recall=1.0
# ============================================
n_negative = (y_true == 0).sum()

results_q4 = []
for i, col in enumerate(model_cols, 1):
    y_pred = df[col].values
    positive_preds = y_pred[y_true == 1]
    t = positive_preds.min()
    negative_preds = y_pred[y_true == 0]
    fp_count = (negative_preds >= t).sum()
    fpr = fp_count / n_negative
    results_q4.append((i, t, fpr))

best_model_q4 = min(results_q4, key=lambda x: x[2])
q4_answer1 = best_model_q4[0]
q4_answer2 = round(best_model_q4[2], 2)
print(f"Q4: model={q4_answer1}, FPR={q4_answer2}")

# ============================================
# Q5: Кросс-корреляция с лагом
# ============================================
satellites = df['satellite_id'].unique()
lag_corrs = {lag: [] for lag in range(-5, 6)}

for sat_id in satellites:
    sat_data = df[df['satellite_id'] == sat_id].sort_values('orbit_number').reset_index(drop=True)
    solar = sat_data['solar_panel_temp_mean'].values
    battery = sat_data['battery_temp_mean'].values
    n = len(solar)
    
    for lag in range(-5, 6):
        if lag > 0:
            s = solar[:n-lag]
            b = battery[lag:]
        elif lag < 0:
            s = solar[-lag:]
            b = battery[:n+lag]
        else:
            s = solar
            b = battery
        
        if len(s) >= 2:
            corr = np.corrcoef(s, b)[0, 1]
            if not np.isnan(corr):
                lag_corrs[lag].append(corr)

avg_corrs = {lag: np.mean(lag_corrs[lag]) for lag in lag_corrs if lag_corrs[lag]}
max_lag = max(avg_corrs, key=lambda x: avg_corrs[x])
q5_answer1 = max_lag
q5_answer2 = round(avg_corrs[max_lag], 2)
print(f"Q5: lag={q5_answer1}, corr={q5_answer2}")

# ============================================
# Q6: Частная корреляция
# ============================================
def partial_correlation(x, y, z):
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]
    numerator = r_xy - r_xz * r_yz
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denominator == 0:
        return 0
    return numerator / denominator

X1 = df['solar_panel_temp_mean'].values
Y1 = df['battery_current_mean'].values
Z1 = df['battery_voltage_mean'].values
pc1 = partial_correlation(X1, Y1, Z1)

X2 = df['battery_current_mean'].values
Y2 = df['battery_voltage_mean'].values
Z2 = df['solar_panel_temp_mean'].values
pc2 = partial_correlation(X2, Y2, Z2)

q6_answer = round(max(abs(pc1), abs(pc2)), 2)
print(f"Q6: {q6_answer}")

# ============================================
# Q7: Максимальный F1-score линейного индекса риска
# ============================================
n_pos = y_true.sum()
X1_norm = df['battery_temp_mean'].values / 40.0
X2_norm = df['reaction_wheel_current_mean'].values / 2.0
X3_norm = df['attitude_error_mean'].values / 1.5

def compute_f1_optimized(y_true, risk_index, n_pos):
    sorted_idx = np.argsort(risk_index)
    sorted_y = y_true[sorted_idx]
    best_f1 = 0
    tp = 0
    fp = 0
    for i in range(len(sorted_y) - 1, -1, -1):
        if sorted_y[i] == 1:
            tp += 1
        else:
            fp += 1
        if tp > 0 and tp + fp > 0:
            precision = tp / (tp + fp)
            recall = tp / n_pos
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
    return best_f1

best_f1 = 0
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1 + 0.001, 0.05):
        w3 = round(1.0 - w1 - w2, 10)
        if w3 < 0:
            continue
        risk_index = w1 * X1_norm + w2 * X2_norm + w3 * X3_norm
        f1 = compute_f1_optimized(y_true, risk_index, n_pos)
        if f1 > best_f1:
            best_f1 = f1

# Уточненный поиск вокруг лучшего
for w1 in np.arange(0.15, 0.25, 0.01):
    for w2 in np.arange(0.55, 0.65, 0.01):
        w3 = round(1.0 - w1 - w2, 10)
        if w3 < 0 or w3 > 0.3:
            continue
        risk_index = w1 * X1_norm + w2 * X2_norm + w3 * X3_norm
        f1 = compute_f1_optimized(y_true, risk_index, n_pos)
        if f1 > best_f1:
            best_f1 = f1

q7_answer = round(best_f1, 3)
print(f"Q7: {q7_answer}")

# ============================================
# Q8: Корреляция доли аномальных витков с количеством отказов
# ============================================
anomaly_fractions = []
failure_counts = []

for sat_id in satellites:
    sat_data = df[df['satellite_id'] == sat_id]
    rwc_std = sat_data['reaction_wheel_current_std'].values
    mu = np.mean(rwc_std)
    sigma = np.std(rwc_std, ddof=0)
    threshold = mu + 3 * sigma
    anomaly_fraction = (rwc_std > threshold).sum() / len(rwc_std)
    failure_count = sat_data['failure_in_12h'].sum()
    anomaly_fractions.append(anomaly_fraction)
    failure_counts.append(failure_count)

q8_answer = round(np.corrcoef(anomaly_fractions, failure_counts)[0, 1], 3)
print(f"Q8: {q8_answer}")

# ============================================
# Q9: Отношение стандартных отклонений остатков регрессии
# ============================================
X_reg = df[['attitude_error_mean', 'reaction_wheel_current_mean']].values
y_reg = df['battery_temp_mean'].values

model = LinearRegression()
model.fit(X_reg, y_reg)
y_pred_reg = model.predict(X_reg)
residuals = y_reg - y_pred_reg

failure_mask = df['failure_in_12h'] == 1
no_failure_mask = df['failure_in_12h'] == 0

std_failure = np.std(residuals[failure_mask], ddof=0)
std_no_failure = np.std(residuals[no_failure_mask], ddof=0)

q9_answer = round(std_failure / std_no_failure, 2)
print(f"Q9: {q9_answer}")

# ============================================
# Q10: Оптимальный ансамбль
# ============================================
orbit_types = df['orbit_type'].values
model_preds_dict = {i: df[col].values for i, col in enumerate(model_cols, 1)}
n_total = len(y_true)

orbit_groups = {0: orbit_types == 0, 1: orbit_types == 1, 2: orbit_types == 2}
n_by_orbit = {ot: mask.sum() for ot, mask in orbit_groups.items()}

def check_constraints(y_pred, y_true, orbit_groups, n_by_orbit, n_total):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    
    if TP + FN > 0 and FN / (TP + FN) > 0.10:
        return False
    if (TP + FP) / n_total > 0.25:
        return False
    
    for ot, mask in orbit_groups.items():
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        TP_g = ((y_pred_g == 1) & (y_true_g == 1)).sum()
        FP_g = ((y_pred_g == 1) & (y_true_g == 0)).sum()
        FN_g = ((y_pred_g == 0) & (y_true_g == 1)).sum()
        
        if TP_g + FN_g > 0 and FN_g / (TP_g + FN_g) > 0.15:
            return False
        if (TP_g + FP_g) / n_by_orbit[ot] > 0.30:
            return False
    return True

best_loss = float('inf')
best_config = None

for combo in [(7, 9, 10), (5, 9, 10), (4, 9, 10), (5, 7, 10), (4, 7, 10), (4, 5, 10)]:
    a, b, c = combo
    pa, pb, pc = model_preds_dict[a], model_preds_dict[b], model_preds_dict[c]
    
    for w1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for w2 in [0.1, 0.2, 0.3, 0.4, 0.5]:
            w3 = 1.0 - w1 - w2
            if w3 < 0.1 or w3 > 0.8:
                continue
            
            p_ensemble = w1 * pa + w2 * pb + w3 * pc
            
            for pct in np.arange(75, 98, 0.5):
                t = np.percentile(p_ensemble, pct)
                y_pred = (p_ensemble > t).astype(int)
                
                if not check_constraints(y_pred, y_true, orbit_groups, n_by_orbit, n_total):
                    continue
                
                TP = ((y_pred == 1) & (y_true == 1)).sum()
                FP = ((y_pred == 1) & (y_true == 0)).sum()
                FN = ((y_pred == 0) & (y_true == 1)).sum()
                loss = 20_000_000 * FN + 100_000 * FP
                
                if loss < best_loss:
                    best_loss = loss
                    best_config = (combo, w1, w2, w3, t, TP, FP, FN, loss)

if best_config:
    combo, w1, w2, w3, t, TP, FP, FN, loss = best_config
    q10_answer1 = ','.join(map(str, sorted(combo)))
    q10_answer2 = round(t, 2)
    q10_answer3 = int(loss)
    print(f"Q10: models={q10_answer1}, threshold={q10_answer2}, loss={q10_answer3}")

# ============================================
# Сохранение результатов
# ============================================
with open('submission.csv', 'w') as f:
    f.write("question_id,answer_1,answer_2,answer_3\n")
    f.write(f'1,"{q1_answer}",,\n')
    f.write(f'2,"{q2_answer1}","{q2_answer2}",\n')
    f.write(f'3,{q3_answer},,\n')
    f.write(f'4,{q4_answer1},{q4_answer2},\n')
    f.write(f'5,{q5_answer1},{q5_answer2},\n')
    f.write(f'6,{q6_answer},,\n')
    f.write(f'7,{q7_answer},,\n')
    f.write(f'8,{q8_answer},,\n')
    f.write(f'9,{q9_answer},,\n')
    f.write(f'10,"{q10_answer1}",{q10_answer2},{q10_answer3}\n')

print("\nResults saved to submission.csv")
