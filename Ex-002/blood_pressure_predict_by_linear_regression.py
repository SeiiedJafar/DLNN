import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


#               محاسبه ضرایب با روش من
# محاسبه میانگین اعداد
def mean(numbers_list, power):
    if power == 2:
        square_numbers_mean = [x ** 2 for x in numbers_list]
        mean = sum(square_numbers_mean) / len(square_numbers_mean)
        return mean
    elif power == 1:
        mean = sum(numbers_list) / len(numbers_list)
        return mean
    else:
        return None


# محاسبه مجموع میانگین اعداد
def array_sigma(array, mean_value, power):
    if power == 2:
        new_array = [(x - mean_value) ** 2 for x in array]
        mean_total = sum(new_array)
        return mean_total
    elif power == 1:
        new_array = [(x - mean_value) for x in array]
        mean_total = sum(new_array)
        return mean_total
    else:
        return None


def arrays_multiplication_sigma(first_array, first_array_mean, second_array, second_array_mean):
    if len(first_array) == len(second_array):
        array_one = [(x - first_array_mean) for x in first_array]
        array_two = [(x - second_array_mean) for x in second_array]
        total = sum(x * y for x, y in zip(array_one, array_two))
        return total
    else:
        return None


df = pd.read_csv('heart_attack_prediction_dataset.csv')

cholesterol = df['Cholesterol'].values.reshape(-1, 1)
exercise = df['Exercise Hours Per Week'].values.reshape(-1, 1)
blood_pressure = [float(item.replace('/', '.')) for item in df['Blood Pressure']]

# ماتریس های پیش بینی من
A = [[array_sigma(cholesterol, mean(cholesterol, 1), 2)[0],
      arrays_multiplication_sigma(cholesterol, mean(cholesterol, 1), exercise, mean(exercise, 1))[0], 0],
     [arrays_multiplication_sigma(cholesterol, mean(cholesterol, 1), exercise, mean(exercise, 1))[0],
      array_sigma(exercise, mean(exercise, 1), 2)[0], 0],
     [0, 0, 1]]
C = [arrays_multiplication_sigma(cholesterol, mean(cholesterol, 1), blood_pressure, mean(blood_pressure, 1))[0],
     arrays_multiplication_sigma(exercise, mean(exercise, 1), blood_pressure, mean(blood_pressure, 1))[0], 0]

# معکوس سازی ماتریس پیش بینی
A_inv = np.linalg.inv(A)

# ماتریس ضرایب
B = np.dot(A_inv, C)
[a, b, c] = [B[0], B[1], B[2]]
print('my predicted c is:')
print(mean(blood_pressure,1)-a*mean(cholesterol,1)-b*mean(exercise,1))
#               محاسبه ضرایب با sklearn

# ایجاد یک مدل رگرسیون خطی
model = LinearRegression()

# آموزش مدل با داده‌ها
model.fit(np.hstack((cholesterol, exercise)), blood_pressure)

# # پیش‌بینی با استفاده از مدل آموزش دیده شده
sklearn_predicted_bP = model.predict(np.hstack((cholesterol, exercise)))
sklearn_predicted_bP = [round(value, 4) for value in sklearn_predicted_bP]

# covariance by sklearn
[e, f, j] = [model.coef_[0], model.coef_[1], model.intercept_]

predicted_bp = [(a * x) + (b * y) + c for x, y in zip(cholesterol, exercise)]
predicted_bp = [round(value[0], 4) for value in predicted_bp]
predicted_bp_with_c = [round(i + j,4) for i in predicted_bp]

print('my covariance prediction')
print(a, ',', b, ',', c, end='\n\n')

print('sklearn covariance prediction')
print(e, ',', f, ',', j, end='\n\n')

print('primary blood pressure is:')
print(blood_pressure[:10], end='\n\n')

print('my predicted blood pressure without c is:')
print(predicted_bp[:10], end='\n\n')

print('my predicted blood pressure with c is:')
print(predicted_bp_with_c[:10], end='\n\n')

print('sklearn predicted blood pressure is:')
print(sklearn_predicted_bP[:10], end='\n\n')
