import numpy as np
from scipy.stats import t

# Данные из таблицы
data = np.array([
    [-1.607, 0.251, -1.414, 1.230, -1.652, -1.406, -1.703, -0.877, -0.953, 0.146],
    [-1.250, -2.651, -1.155, -2.489, -0.875, -0.395, -1.025, -2.327, -1.400, -0.273],
    [-0.562, -1.833, -3.236, -1.872, -2.233, -0.596, -0.632, -1.496, -1.106, -0.669],
    [-0.572, -1.609, -1.290, -2.054, -1.001, -2.211, -1.142, -0.232, -2.330, -2.071],
    [-2.305, -0.952, -0.862, -0.366, -1.493, 0.303, -0.295, -0.794, -1.581, -2.629]
])

# Разделение на две выборки
sample1 = data[:2, :].flatten()  # первые 20 элементов (2 строки по 10 элементов)
sample2 = data[2:, :].flatten()  # следующие 30 элементов (3 строки по 10 элементов)


# Функция проверки гипотезы о равенстве средних
def check_means_equal(sample1, sample2, e=0.07):
    n, m = len(sample1), len(sample2)

    # средние значения выборок
    mean1, mean2 = np.mean(sample1), np.mean(sample2)

    # выборочные дисперсии
    var1, var2 = np.var(sample1, ddof=0), np.var(sample2, ddof=0)

    # объединенное стандартное отклонение
    pooled_std = np.sqrt((n * var1 + m * var2) / (n + m - 2))

    # Стандартизированная разность средних
    t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(1 / n + 1 / m))

    # Степени свободы
    df = n + m - 2

    # Критическое значение для заданного уровня значимости
    critical_value = t.ppf(1 - e/2, df)

    # Принятие решения
    if abs(t_stat) > critical_value:
        result = "Отвергаем нулевую гипотезу: средние не равны."
    else:
        result = "Не отвергаем нулевую гипотезу: средние равны."
    print(result)
    return t_stat, critical_value


# Проверка гипотезы о равенстве средних
t_stat, critical_value = check_means_equal(sample1, sample2)

# Вывод результатов
print(f"t-statistic: {t_stat}")
print(f"Critical Value: {critical_value}")