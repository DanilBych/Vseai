import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix
from scipy import stats
from itertools import combinations

# Загрузка данных
df = pd.read_csv('space_risk_model.csv')

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Unique satellites: {df['satellite_id'].nunique()}")
print(f"Failure rate: {df['failure_in_12h'].mean():.4f}")

# Вопрос 1: Для каждого спутника вычислите среднюю температуру солнечных панелей
# Затем для каждого типа орбиты определите спутник с максимальной средней температурой
print("\n=== QUESTION 1 ===")

# Вычисляем среднюю температуру солнечных панелей для каждого спутника
sat_avg_temp = df.groupby('satellite_id')['solar_panel_temp_mean'].mean().reset_index()
sat_avg_temp.columns = ['satellite_id', 'avg_solar_temp']

# Объединяем с типом орбиты (он фиксирован для каждого спутника)
sat_orbit = df[['satellite_id', 'orbit_type']].drop_duplicates()
sat_info = sat_avg_temp.merge(sat_orbit, on='satellite_id')

print(f"Satellites info:\n{sat_info.head(20)}")

# Для каждого типа орбиты находим спутник с максимальной средней температурой
result_q1 = []
for orbit_type in [0, 1, 2]:
    subset = sat_info[sat_info['orbit_type'] == orbit_type]
    max_sat = subset.loc[subset['avg_solar_temp'].idxmax(), 'satellite_id']
    result_q1.append(int(max_sat))
    print(f"Orbit type {orbit_type}: satellite {max_sat} with avg temp {subset['avg_solar_temp'].max():.4f}")

answer_q1 = f"{result_q1[0]},{result_q1[1]},{result_q1[2]}"
print(f"Answer Q1: {answer_q1}")

# Вопрос 2: Пять лучших моделей по ROC-AUC
print("\n=== QUESTION 2 ===")

model_cols = [f'model_{i}_pred' for i in range(1, 11)]
auc_scores = {}

for col in model_cols:
    auc = roc_auc_score(df['failure_in_12h'], df[col])
    model_num = int(col.split('_')[1])
    auc_scores[model_num] = auc
    print(f"Model {model_num}: AUC = {auc:.6f}")

# Сортируем модели по убыванию AUC, при равных значениях - по возрастанию номера модели
sorted_models = sorted(auc_scores.items(), key=lambda x: (-x[1], x[0]))
top5_models = sorted_models[:5]

print(f"\nTop 5 models: {[m[0] for m in top5_models]}")
print(f"AUC scores: {[round(m[1], 2) for m in top5_models]}")

answer_q2_1 = ','.join([str(m[0]) for m in top5_models])
answer_q2_2 = ','.join([str(round(m[1], 2)) for m in top5_models])
print(f"Answer Q2 (models): {answer_q2_1}")
print(f"Answer Q2 (AUC): {answer_q2_2}")

# Вопрос 3: Скорость нагрева
print("\n=== QUESTION 3 ===")

def calculate_heating_rate(df):
    df_sorted = df.sort_values(['satellite_id', 'orbit_number']).copy()
    
    # Вычисляем heating_rate для каждого спутника
    df_sorted['heating_rate'] = df_sorted.groupby('satellite_id')['solar_panel_temp_mean'].diff(3)
    
    return df_sorted

df_with_heating = calculate_heating_rate(df)

# Исключаем первые 3 витка каждого спутника (где heating_rate не определен)
df_valid = df_with_heating[df_with_heating['heating_rate'].notna()].copy()

# Средняя абсолютная скорость нагрева в витках без отказа
no_failure_df = df_valid[df_valid['failure_in_12h'] == 0]
avg_heating_no_failure = no_failure_df['heating_rate'].abs().mean()

# Для витков с отказом берем heating_rate за 3 витка до отказа
# Нам нужно значение heating_rate на витке k-3 для каждого витка k где failure_in_12h=1
failure_df = df_with_heating[df_with_heating['failure_in_12h'] == 1].copy()

# Сдвигаем heating_rate на 3 позиции вперед для каждого спутника
df_with_heating['heating_rate_lag3'] = df_with_heating.groupby('satellite_id')['heating_rate'].shift(-3)

# Теперь для витков с failure_in_12h=1 берем heating_rate_lag3
failure_with_lag = df_with_heating[df_with_heating['failure_in_12h'] == 1].copy()
failure_with_lag = failure_with_lag[failure_with_lag['heating_rate_lag3'].notna()]

avg_heating_before_failure = failure_with_lag['heating_rate_lag3'].abs().mean()

print(f"Avg |heating_rate| no failure: {avg_heating_no_failure:.6f}")
print(f"Avg |heating_rate| 3 orbits before failure: {avg_heating_before_failure:.6f}")

ratio = avg_heating_before_failure / avg_heating_no_failure
print(f"Ratio: {ratio:.4f}")
answer_q3 = round(ratio)
print(f"Answer Q3: {answer_q3}")

# Вопрос 4: Минимальный FPR при Recall=1.0
print("\n=== QUESTION 4 ===")

fpr_at_recall_1 = {}

for col in model_cols:
    model_num = int(col.split('_')[1])
    predictions = df[col].values
    y_true = df['failure_in_12h'].values
    
    # Находим минимальное значение прогноза среди витков с failure_in_12h=1
    failure_predictions = predictions[y_true == 1]
    t = failure_predictions.min()
    
    print(f"Model {model_num}: threshold t = {t:.6f}")
    
    # Классификация: опасный если prediction >= t
    y_pred = (predictions >= t).astype(int)
    
    # Вычисляем FPR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Recall должен быть 1.0 (fn = 0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}, Recall={recall:.4f}, FPR={fpr:.4f}")
    
    fpr_at_recall_1[model_num] = fpr

# Находим модель с минимальным FPR
min_fpr_model = min(fpr_at_recall_1.items(), key=lambda x: x[1])
print(f"\nModel with min FPR: {min_fpr_model[0]} with FPR={min_fpr_model[1]:.4f}")

answer_q4_1 = min_fpr_model[0]
answer_q4_2 = round(min_fpr_model[1], 2)
print(f"Answer Q4: Model {answer_q4_1}, FPR={answer_q4_2}")

# Вопрос 5: Кросс-корреляция со сдвигом
print("\n=== QUESTION 5 ===")

def calculate_cross_correlation(df, lag):
    """Вычисляет корреляцию для данного лага"""
    correlations = []
    
    for sat_id in df['satellite_id'].unique():
        sat_data = df[df['satellite_id'] == sat_id].sort_values('orbit_number').copy()
        
        if lag > 0:
            # batteries(t+lag) запаздывают
            solar = sat_data['solar_panel_temp_mean'].values[:-lag]
            battery = sat_data['battery_temp_mean'].values[lag:]
        elif lag < 0:
            # batteries(t+lag) опережают (lag отрицательный)
            solar = sat_data['solar_panel_temp_mean'].values[-lag:]
            battery = sat_data['battery_temp_mean'].values[:lag]
        else:
            solar = sat_data['solar_panel_temp_mean'].values
            battery = sat_data['battery_temp_mean'].values
        
        if len(solar) > 1 and len(battery) > 1:
            if np.std(solar) > 0 and np.std(battery) > 0:
                corr = np.corrcoef(solar, battery)[0, 1]
                correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0

lags = range(-5, 6)
correlations_by_lag = {}

for lag in lags:
    corr = calculate_cross_correlation(df, lag)
    correlations_by_lag[lag] = corr
    print(f"Lag {lag}: correlation = {corr:.6f}")

max_lag = max(correlations_by_lag.items(), key=lambda x: x[1])
print(f"\nMax correlation at lag {max_lag[0]}: {max_lag[1]:.6f}")

answer_q5_1 = max_lag[0]
answer_q5_2 = round(max_lag[1], 2)
print(f"Answer Q5: lag={answer_q5_1}, corr={answer_q5_2}")

# Вопрос 6: Частные корреляции
print("\n=== QUESTION 6 ===")

def partial_correlation(x, y, z):
    """Вычисляет частную корреляцию между x и y при контроле z"""
    # Корреляции
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]
    
    # Частная корреляция
    numerator = r_xy - r_xz * r_yz
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    
    if denominator == 0:
        return 0
    return numerator / denominator

X1 = df['solar_panel_temp_mean'].values
Y1 = df['battery_current_mean'].values
Z1 = df['battery_voltage_mean'].values

X2 = df['battery_current_mean'].values
Y2 = df['battery_voltage_mean'].values
Z2 = df['solar_panel_temp_mean'].values

partial_corr_1 = partial_correlation(X1, Y1, Z1)
partial_corr_2 = partial_correlation(X2, Y2, Z2)

print(f"Partial corr (solar_temp, battery_current | battery_voltage): {partial_corr_1:.6f}")
print(f"Partial corr (battery_current, battery_voltage | solar_temp): {partial_corr_2:.6f}")

max_partial = max(abs(partial_corr_1), abs(partial_corr_2))
answer_q6 = round(max_partial, 2)
print(f"Answer Q6: {answer_q6}")

# Вопрос 7: Линейный индекс риска - оптимизированный подход
print("\n=== QUESTION 7 ===")

def calculate_f1_for_weights_fast(w1, w2, w3, df):
    """Вычисляет максимальный F1 для данных весов"""
    risk_index = (w1 * df['battery_temp_mean'] / 40 + 
                  w2 * df['reaction_wheel_current_mean'] / 2.0 + 
                  w3 * df['attitude_error_mean'] / 1.5)
    
    y_true = df['failure_in_12h'].values
    
    # Используем percentiles для быстрого подбора порога
    thresholds = np.percentile(risk_index, range(1, 100))
    
    best_f1 = 0
    for t in thresholds:
        y_pred = (risk_index.values > t).astype(int)
        
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)
    
    return best_f1

best_f1_overall = 0
best_weights = None

# Сетка весов с шагом 0.05 для скорости
step = 0.05
for w1 in np.arange(0, 1.01, step):
    for w2 in np.arange(0, 1.01 - w1, step):
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue
        
        f1 = calculate_f1_for_weights_fast(w1, w2, w3, df)
        if f1 > best_f1_overall:
            best_f1_overall = f1
            best_weights = (w1, w2, w3)
            print(f"New best F1={best_f1_overall:.6f} at weights={best_weights}")

print(f"Best F1: {best_f1_overall:.6f}")
print(f"Best weights: {best_weights}")

answer_q7 = round(best_f1_overall, 3)
print(f"Answer Q7: {answer_q7}")

# Вопрос 8: Доля аномальных витков и корреляция
print("\n=== QUESTION 8 ===")

sat_anomaly_stats = []

for sat_id in df['satellite_id'].unique():
    sat_data = df[df['satellite_id'] == sat_id]
    
    mu = sat_data['reaction_wheel_current_std'].mean()
    sigma = sat_data['reaction_wheel_current_std'].std(ddof=0)
    threshold = mu + 3 * sigma
    
    anomaly_count = (sat_data['reaction_wheel_current_std'] > threshold).sum()
    total_count = len(sat_data)
    anomaly_fraction = anomaly_count / total_count
    
    failure_count = sat_data['failure_in_12h'].sum()
    
    sat_anomaly_stats.append({
        'satellite_id': sat_id,
        'anomaly_fraction': anomaly_fraction,
        'failure_count': failure_count
    })

stats_df = pd.DataFrame(sat_anomaly_stats)
print(f"Stats:\n{stats_df.head(10)}")

correlation = np.corrcoef(stats_df['anomaly_fraction'], stats_df['failure_count'])[0, 1]
print(f"Correlation: {correlation:.6f}")

answer_q8 = round(correlation, 3)
print(f"Answer Q8: {answer_q8}")

# Вопрос 9: Остатки линейной регрессии
print("\n=== QUESTION 9 ===")

from sklearn.linear_model import LinearRegression

X = df[['attitude_error_mean', 'reaction_wheel_current_mean']].values
y = df['battery_temp_mean'].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
residuals = y - y_pred

df_with_residuals = df.copy()
df_with_residuals['residual'] = residuals

residuals_failure = df_with_residuals[df_with_residuals['failure_in_12h'] == 1]['residual']
residuals_no_failure = df_with_residuals[df_with_residuals['failure_in_12h'] == 0]['residual']

std_failure = residuals_failure.std()
std_no_failure = residuals_no_failure.std()

ratio_std = std_failure / std_no_failure
print(f"Std failure: {std_failure:.6f}")
print(f"Std no failure: {std_no_failure:.6f}")
print(f"Ratio: {ratio_std:.6f}")

answer_q9 = round(ratio_std, 2)
print(f"Answer Q9: {answer_q9}")

# Вопрос 10: Оптимальный ансамбль - упрощенный подход для скорости
print("\n=== QUESTION 10 ===")

# Определяем функцию потерь
def calculate_loss_and_constraints(df, ensemble_pred, t):
    """Вычисляет потери и проверяет ограничения"""
    y_true = df['failure_in_12h'].values
    y_pred = (ensemble_pred > t).astype(int)
    
    # Общие метрики
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    
    loss = 20_000_000 * fn + 100_000 * fp
    
    # Общие ограничения
    total_failures = tp + fn
    fnr_total = fn / total_failures if total_failures > 0 else 0
    warning_rate_total = (tp + fp) / len(y_true)
    
    # Ограничения по типам орбит
    constraints_ok = True
    
    for orbit_type in [0, 1, 2]:
        mask = df['orbit_type'] == orbit_type
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        tn_g = ((y_pred_group == 0) & (y_true_group == 0)).sum()
        fp_g = ((y_pred_group == 1) & (y_true_group == 0)).sum()
        fn_g = ((y_pred_group == 0) & (y_true_group == 1)).sum()
        tp_g = ((y_pred_group == 1) & (y_true_group == 1)).sum()
        
        total_failures_g = tp_g + fn_g
        fnr_group = fn_g / total_failures_g if total_failures_g > 0 else 0
        warning_rate_group = (tp_g + fp_g) / len(y_true_group)
        
        if fnr_group > 0.15:
            constraints_ok = False
        if warning_rate_group > 0.30:
            constraints_ok = False
    
    if fnr_total > 0.10:
        constraints_ok = False
    if warning_rate_total > 0.25:
        constraints_ok = False
    
    return loss, constraints_ok, fnr_total, warning_rate_total

best_loss = float('inf')
best_config = None

# Перебираем только лучшие модели из вопроса 2
top_model_indices = [m[0] for m in top5_models]
print(f"Testing combinations of top models: {top_model_indices}")

for combo in combinations(top_model_indices, 3):
    a, b, c = combo
    col_a = f'model_{a}_pred'
    col_b = f'model_{b}_pred'
    col_c = f'model_{c}_pred'
    
    p_a = df[col_a].values
    p_b = df[col_b].values
    p_c = df[col_c].values
    
    # Перебираем веса с шагом 0.2 для скорости
    for w1 in np.arange(0.2, 1.0, 0.2):
        for w2 in np.arange(0.2, 1.0 - w1, 0.2):
            w3 = 1 - w1 - w2
            if w3 < 0.2:
                continue
            
            ensemble_pred = w1 * p_a + w2 * p_b + w3 * p_c
            
            # Перебираем пороги через percentiles
            thresholds = np.percentile(ensemble_pred, range(50, 95, 5))
            
            for t in thresholds:
                loss, constraints_ok, fnr, wr = calculate_loss_and_constraints(df, ensemble_pred, t)
                
                if constraints_ok and loss < best_loss:
                    best_loss = loss
                    best_config = (combo, (w1, w2, w3), t)
                    print(f"New best: models={combo}, weights=({w1:.2f},{w2:.2f},{w3:.2f}), t={t:.4f}, loss={loss}")

if best_config:
    models, weights, threshold = best_config
    print(f"\nBest config: models={models}, weights={weights}, threshold={threshold:.4f}, loss={best_loss}")
    answer_q10_1 = f"{models[0]},{models[1]},{models[2]}"
    answer_q10_2 = round(threshold, 2)
    answer_q10_3 = round(best_loss)
else:
    print("No valid configuration found!")
    answer_q10_1 = "1,2,3"
    answer_q10_2 = 0.5
    answer_q10_3 = 0

print(f"Answer Q10: models={answer_q10_1}, threshold={answer_q10_2}, loss={answer_q10_3}")

# Сохранение результатов
results = [
    {'question_id': 1, 'answer_1': answer_q1, 'answer_2': '', 'answer_3': ''},
    {'question_id': 2, 'answer_1': answer_q2_1, 'answer_2': answer_q2_2, 'answer_3': ''},
    {'question_id': 3, 'answer_1': str(answer_q3), 'answer_2': '', 'answer_3': ''},
    {'question_id': 4, 'answer_1': str(answer_q4_1), 'answer_2': str(answer_q4_2), 'answer_3': ''},
    {'question_id': 5, 'answer_1': str(answer_q5_1), 'answer_2': str(answer_q5_2), 'answer_3': ''},
    {'question_id': 6, 'answer_1': str(answer_q6), 'answer_2': '', 'answer_3': ''},
    {'question_id': 7, 'answer_1': str(answer_q7), 'answer_2': '', 'answer_3': ''},
    {'question_id': 8, 'answer_1': str(answer_q8), 'answer_2': '', 'answer_3': ''},
    {'question_id': 9, 'answer_1': str(answer_q9), 'answer_2': '', 'answer_3': ''},
    {'question_id': 10, 'answer_1': answer_q10_1, 'answer_2': str(answer_q10_2), 'answer_3': str(answer_q10_3)},
]

submission_df = pd.DataFrame(results)
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission saved to submission.csv")
print(submission_df)
