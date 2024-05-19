import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, uniform
from scipy.special import kolmogorov

# Заданные данные
data = [
    0.332, 0.927, 0.692, 0.768, 0.151, 0.121, 0.634, 0.647, 0.890, 0.035,
    0.752, 0.357, 0.934, 0.176, 0.306, 0.587, 0.413, 0.359, 0.933, 0.067,
    0.458, 0.412, 0.355, 0.050, 0.303, 0.107, 0.205, 0.382, 0.333, 0.850
]

data = np.array(data)


# Функция для вычисления эмпирической функции распределения
def ecdf(data):
    x = np.sort(data)  # сортируем
    y = np.arange(1, len(data) + 1) / len(data)  # значение эмпирической функции распределения на каждой из точек данных
    return x, y


x, y = ecdf(data)

# Построение графика
plt.figure(figsize=(10, 6))
plt.step(x, y, where='post', label='Empirical CDF')

# Теоретическая функция распределения для равномерного распределения
plt.plot(x, uniform.cdf(x), label='Theoretical CDF (Uniform)', linestyle='--')

plt.xlabel('Value')
plt.ylabel('ECDF')
plt.title('Empirical CDF vs Theoretical CDF')
plt.legend()
plt.grid(True)
plt.show()


# Реализация критерия Колмогорова
def kolmogorov_test(data):
    n = len(data)
    x, y = ecdf(data)

    # Максимальное положительное отклонение между ECDF и теоретической CDF
    d_plus = np.max(y - uniform.cdf(x))

    # Максимальное отрицательное отклонение между ECDF и теоретической CDF
    d_minus = np.max(uniform.cdf(x) - (y - 1 / n))

    # Максимальное отклонение
    d = max(d_plus, d_minus)
    return d


# Максимальное отклонение между эмпирической и теоретической функцией распределения
d_statistic = kolmogorov_test(data)

# Нормализация статистики
n = len(data)
lambda_stat = np.sqrt(n) * d_statistic

# Вычисление критического значения
alpha = 0.05
critical_value = 1.36 / np.sqrt(n)

# Вычисление p-value с использованием распределения Колмогорова
p_value_kolm = 1 - kolmogorov(lambda_stat)

# Вывод результатов на русском языке
print(f"Статистика Колмогорова: {d_statistic}")
print(f"Статистика Лямбда: {lambda_stat}")
print(f"Критическое значение (для уровня значимости α = 0.05): {critical_value}")
print(f"P-значение (Колмогоров): {p_value_kolm}")

# если вычисленная статистика D превышает критическое значение, мы отвергаем нулевую гипотезу о равномерном распределении
if d_statistic > critical_value:
    print("Отвергаем нулевую гипотезу о равномерности распределения (d_statistic > critical_value)")
else:
    print("Нет оснований отвергнуть нулевую гипотезу о равномерности распределения (d_statistic <= critical_value)")

# # Проверка гипотезы о том, что данные data следуют равномерному распределению с помощью критерия Колмогорова-Смирнова
# ks_statistic, p_value_ks = kstest(data, 'uniform')
#
# print(f"KS Статистика: {ks_statistic}")
# print(f"P-значение (KS): {p_value_ks}")
