
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn

# Set the seaborn style
sns.set_theme(style="whitegrid")

# --- الجزء الأول: اشتقاق دالة cos(x) باستخدام الفرق الأمامي ومقارنة النتيجة بالحل الدقيق ---

# تعريف حجم الخطوة h
h = 0.1

# إنشاء شبكة x من 0 إلى 2π بفواصل h
x = np.arange(0, 2 * np.pi, h)

# حساب قيم cos(x)
y = np.cos(x)

# حساب الفرق الأمامي (تقريب المشتقة)
forward_diff = np.diff(y) / h

# شبكة x جديدة تناسب طول الفرق الأمامي
x_diff = x[:-1]

# الحل الدقيق للمشتقة وهو -sin(x)
exact_solution = -np.sin(x_diff)

# رسم النتائج
plt.figure(figsize=(12, 8))
plt.plot(x_diff, forward_diff, '--', label='Finite difference approximation')
plt.plot(x_diff, exact_solution, label='Exact solution')
plt.legend()
plt.title("Forward Difference Approximation vs Exact Derivative")
plt.xlabel("x")
plt.ylabel("Derivative")
plt.grid(True)
plt.show()

# حساب الخطأ الأقصى
max_error = max(abs(exact_solution - forward_diff))
print("Maximum error:", max_error)

# --- الجزء الثاني: تقريب الاشتقاق عند نقطة محددة باستخدام قيم مختلفة لـ h ---

# النقطة التي نحسب عندها المشتقة
x0 = 0.7

# خطوات صغيرة h = 2^(-1), 2^(-2), ..., 2^(-29)
h = 2.**-np.arange(1, 30)

# تقريب المشتقة باستخدام الفرق الأمامي
df = (np.cos(x0 + h) - np.cos(x0)) / h

# القيمة الحقيقية للمشتقة عند x0
true_value = -np.sin(x0)

# طباعة النتائج
print("\nk | Approximation        | Ratio of errors    | Relative differences")
print("---|----------------------|--------------------|----------------------")

previous_approximation = None
previous_error = None

for k in range(1, len(h) + 1):
    approximation = df[k - 1]
    error = np.abs(approximation - true_value)
    
    ratio = np.abs(previous_error / error) if previous_error is not None else " "
    relative_difference = (
        np.abs((approximation - previous_approximation) / previous_approximation)
        if previous_approximation is not None else " "
    )

    formatted_approximation = f"{approximation:.15f}"
    formatted_ratio = f"{ratio:.6f}" if isinstance(ratio, float) else ratio
    formatted_relative_difference = f"{relative_difference:.10f}" if isinstance(relative_difference, float) else relative_difference

    print(f"{k:<3}| {formatted_approximation:<22}| {formatted_ratio:<20}| {formatted_relative_difference:<20}")

    previous_error = error
    previous_approximation = approximation

# طباعة القيمة الحقيقية
print(f"\nTrue value of the derivative: {true_value:.17f}")
