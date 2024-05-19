import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

# Исходные данные
data = np.array([
    0.332, 0.927, 0.692, 0.768, 0.151, 0.121, 0.634, 0.647, 0.890, 0.035,
    0.752, 0.357, 0.934, 0.176, 0.306, 0.587, 0.413, 0.359, 0.933, 0.067,
    0.458, 0.412, 0.355, 0.050, 0.303, 0.107, 0.205, 0.382, 0.333, 0.850
])

# количество интервалов (k)
k = 5

# наблюдаемые частоты и границы интервалов
observed_freq, bin_edges = np.histogram(data, bins=k)

# ожидаемая частота для равномерного распределения
expected_freq = len(data) / k

# статистика критерия хи-квадрат
chi_squared_statistic = np.sum((observed_freq - expected_freq)**2 / expected_freq)

# степени свободы
df = k - 1

# вычисление p-value
p_value = 1 - chi2.cdf(chi_squared_statistic, df)

# Вывод результатов
print(f"Chi-squared Statistic: {chi_squared_statistic}")
print(f"P-value: {p_value}")
print(f"Degrees of Freedom: {df}")

# Построение графика эмпирической функции распределения
sorted_data = np.sort(data)
y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

plt.step(sorted_data, y_vals, where='post', label='Empirical CDF')
plt.plot([0, 1], [0, 1], 'r--', label='Uniform CDF')
plt.xlabel('Data')
plt.ylabel('CDF')
plt.title('Empirical CDF vs. Uniform CDF')
plt.legend()
plt.show()
