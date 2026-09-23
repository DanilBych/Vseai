import pandas as pd
import numpy as np
from itertools import combinations

# Загрузка данных
df = pd.read_csv('/workspace/space_risk_model.csv')
print(f"Shape: {df.shape}")

# Стоимость ошибок
C_FN = 20_000_000
C_FP = 100_000

# Ограничения
MAX_FNR_TOTAL = 0.10
MAX_ALERT_TOTAL = 0.25
MAX_FNR_GROUP = 0.15
MAX_ALERT_GROUP = 0.30

model_preds_arr = np.column_stack([df[f'model_{i}_pred'].values for i in range(1, 11)])
y_true = df['failure_in_12h'].values.astype(np.int32)
orbit_types = df['orbit_type'].values.astype(np.int32)
N_total = len(df)

# Группы по типу орбиты
group_masks = [(orbit_types == k) for k in [0, 1, 2]]
N_groups = np.array([mask.sum() for mask in group_masks])
FN_groups_ref = np.array([y_true[mask].sum() for mask in group_masks])

print(f"N_total: {N_total}")
print(f"N_groups: {N_groups}")
print(f"FN_groups_ref: {FN_groups_ref}")

total_pos_all = y_true.sum()
print(f"Total positives: {total_pos_all}")

# Перебор всех комбинаций из 3 моделей
best_loss = float('inf')
best_solution = None

model_combinations = list(combinations(range(10), 3))
print(f"Number of model combinations: {len(model_combinations)}")

# Для оптимизации переберем веса с шагом 0.25 сначала
weight_steps = [0.25, 0.5, 0.75]

counter = 0
for combo in model_combinations:
    counter += 1
    if counter % 30 == 0:
        print(f"Processing combination {counter}/{len(model_combinations)}")
    
    a, b, c = combo
    p_a = model_preds_arr[:, a]
    p_b = model_preds_arr[:, b]
    p_c = model_preds_arr[:, c]
    
    # Перебор весов
    for w1 in weight_steps:
        for w2 in weight_steps:
            w3 = 1 - w1 - w2
            if w3 <= 0 or w1 <= 0 or w2 <= 0:
                continue
            
            # Ансамблевый прогноз
            p_ensemble = w1 * p_a + w2 * p_b + w3 * p_c
            
            # Сортируем для быстрого перебора порогов
            sorted_idx = np.argsort(p_ensemble)
            sorted_p = p_ensemble[sorted_idx]
            sorted_y = y_true[sorted_idx]
            sorted_orbit = orbit_types[sorted_idx]
            
            total_pos = sorted_y.sum()
            
            # Быстрый расчет FN и alert для каждого порога
            # cumsum дает TP и FP для каждого порога
            cum_TP_all = np.cumsum(sorted_y)
            cum_FP_all = np.cumsum(1 - sorted_y)
            
            # FN = total_pos - TP
            FN_all = total_pos - cum_TP_all
            
            # FNR = FN / total_pos
            fnr_total_all = FN_all / total_pos
            
            # Alert = (TP + FP) / N_total
            alert_total_all = (cum_TP_all + cum_FP_all) / N_total
            
            # Находим индексы где выполняются общие ограничения
            valid_mask = (fnr_total_all <= MAX_FNR_TOTAL) & (alert_total_all <= MAX_ALERT_TOTAL)
            
            if not valid_mask.any():
                continue
            
            # Теперь проверяем групповые ограничения только для валидных индексов
            valid_indices = np.where(valid_mask)[0]
            
            # Для каждого валидного индекса вычисляем групповые метрики
            for idx in valid_indices:
                # Вычисляем групповые TP и FP до этого индекса
                mask_up_to = np.arange(N_total) <= idx
                group_TP_k = np.zeros(3, dtype=np.int32)
                group_FP_k = np.zeros(3, dtype=np.int32)
                
                for k in range(3):
                    group_mask = (sorted_orbit == k)
                    group_TP_k[k] = ((sorted_y == 1) & group_mask & mask_up_to).sum()
                    group_FP_k[k] = ((sorted_y == 0) & group_mask & mask_up_to).sum()
                
                # Проверяем групповые ограничения
                valid_groups = True
                for k in range(3):
                    group_FN_k = FN_groups_ref[k] - group_TP_k[k]
                    fnr_g = group_FN_k / FN_groups_ref[k] if FN_groups_ref[k] > 0 else 0
                    alert_g = (group_TP_k[k] + group_FP_k[k]) / N_groups[k]
                    if fnr_g > MAX_FNR_GROUP or alert_g > MAX_ALERT_GROUP:
                        valid_groups = False
                        break
                
                if valid_groups:
                    TP = cum_TP_all[idx]
                    FP = cum_FP_all[idx]
                    FN = FN_all[idx]
                    loss = C_FN * FN + C_FP * FP
                    
                    threshold_val = sorted_p[idx]
                    
                    if loss < best_loss or (loss == best_loss and (threshold_val < best_solution[3] if best_solution else True)):
                        best_loss = loss
                        best_solution = (a+1, b+1, c+1, threshold_val, w1, w2, w3, loss)
                        print(f"New best: models={(a+1,b+1,c+1)}, t={threshold_val:.4f}, loss={loss:,}")

if best_solution:
    a, b, c, t, w1, w2, w3, loss = best_solution
    models_sorted = tuple(sorted([a, b, c]))
    print(f"\nBest solution:")
    print(f"Models: {models_sorted}")
    print(f"Threshold: {t:.4f}")
    print(f"Weights: {w1}, {w2}, {w3}")
    print(f"Loss: {loss:,}")
else:
    print("No valid solution found!")
