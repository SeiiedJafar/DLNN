import random
import math
import matplotlib.pyplot as plt
import numpy as np  #برای مقایسه با نتایج کد من


# محاسبه میانگین ضرب عناصر متناظر دو لیست از اعداد حقیقی
def xy_mean(x_numbers, y_numbers):
    if len(x_numbers) == len(y_numbers):
        mean = sum(x * y for x, y in zip(x_numbers, y_numbers)) / len(x_numbers)
        return mean
    else:
        print('x_numbers list length must equal y_numbers list length')


# محاسبه میانگین اعداد
def numbers_mean(numbers_list, power):
    if power == 2:
        square_numbers_mean = [x ** 2 for x in numbers_list]
        mean = sum(square_numbers_mean) / len(square_numbers_mean)
        return mean
    elif power == 1:
        mean = sum(numbers_list) / len(numbers_list)
        return mean
    else:
        print('Power must be 1 or 2 !')


# محاسبه واریانس بدون اصلاح بسل
def variance(list):
    numbers_list_variance = sum((x - numbers_mean(list, 1)) ** 2 for x in list) / (len(list))
    return numbers_list_variance


# محاسبه انحراف معیار بدون اصلاح بسل
def std_deviation(list):
    numbers_list_std_deviation = math.sqrt(variance(list))
    return numbers_list_std_deviation


# محاسبه شیب خط از داده های x,y های اولیه
def slop(x_numbers, y_numbers):
    m = xy_mean(x_numbers, y_numbers) / numbers_mean(x_numbers, 2)
    return m


# محاسبه ضریب همبستگی
def normalized_covariance(x_numbers, y_numbers):
    x_numbers_mean = numbers_mean(x_numbers, 1)
    y_numbers_mean = numbers_mean(y_numbers, 1)
    adjusted_x_numbers = [x - x_numbers_mean for x in x_numbers]
    adjusted_y_numbers = [y - y_numbers_mean for y in y_numbers]
    adjusted_numbers_multiply_mean = xy_mean(adjusted_x_numbers, adjusted_y_numbers)
    correlation_coefficient = adjusted_numbers_multiply_mean / (std_deviation(x_numbers) * std_deviation(y_numbers))
    return correlation_coefficient


# محاسبه انحراف داده های اولیه از داده های بیش بینی شده
def y_deviation(predicted_y_numbers, y_numbers):
    intercepts_list = [y - pred_y for pred_y, y in zip(predicted_y_numbers, y_numbers)]
    intercept_mean = numbers_mean(intercepts_list, 1)
    return intercept_mean


# بیش بینی y ها بر اساس فرمول رگرسیون y بر x
def y_prediction(x_numbers, y_numbers):
    cc = normalized_covariance(x_numbers, y_numbers)
    x_numbers_std_deviation = std_deviation(x_numbers)
    y_numbers_std_deviation = std_deviation(y_numbers)
    y_predicted_numbers = [
        (cc * ((x - numbers_mean(x_numbers, 1)) / x_numbers_std_deviation) * y_numbers_std_deviation) + numbers_mean(
            y_numbers, 1) for x in x_numbers]
    return y_predicted_numbers


# مقدار پیش فرض a
a = 2

# تولید 1000 عدد تصادفی برای b در بازه [-3, 5] به دقت 4 رقم اعشار
b_values = [round(random.uniform(-3, 5), 4) for b_number in range(1000)]

# تولید 1000 عدد تصادفی برای x در بازه [0, 100] به دقت 4 رقم اعشار
x_values = [round(random.uniform(0, 100), 4) for x_number in range(1000)]

# محاسبه y بر اساس معادله y = a(x^2) + b
y_values = [a * x + b for x, b in zip(x_values, b_values)]
y_values = [round(item, 4) for item in y_values]

# تنظیم شعاع نقاط روی نمودار
point_radius = 1

# تست ضریب همبستگی من با ضریب همبستگی محاسبه شده در numpy
correlation_coefficient = normalized_covariance(x_values, y_values)
print(f'my correlation_coefficient is {correlation_coefficient} and')
print(f'numpy correlation_coefficient is {np.corrcoef(x_values, y_values)[0, 1]}')

#محاسبه شیب خط حاصل از داده های اولیه
slop_of_primary_values = round(slop(x_values, y_values), 4)
print(f'calculated slope (in other word a) is {slop_of_primary_values}')

#بیش بینی مقادیر Y مناسب برای رگرسیون
# predicted_y_values = [slop_of_primary_values * x + numbers_mean(b_values, 1) for x in x_values]
predicted_y_values = y_prediction(x_values, y_values)
predicted_y_values = [round(item, 4) for item in predicted_y_values]

# انحراف مقادیر اولیه y و پیش بینی شده Y
predicted_and_primary_y_deviation = y_deviation(y_values, predicted_y_values)
print(f'predicted and primary y deviation {predicted_and_primary_y_deviation}')

# تبدیل لیست‌های x_values و y_values به آرایه‌های NumPy
x_data = np.array(x_values)
y_data = np.array(y_values)

# بدست آوردن شیب پ عرض از مبدا خط فیت با استفاده از پلینوم درجه ۱
coefficients = np.polyfit(x_data, y_data, 1)

# بازیابی ضرایب خط تطابقی
slope = coefficients[0]
print(slope)
intercept = coefficients[1]

# رسم نمودار
plt.scatter(x_values, y_values, alpha=0.2, s=point_radius)  # مثدار alpha شفافیت نقاط است

# بیش بینی y با فرمول رگرسیون
plt.plot(x_values, predicted_y_values, color='red')

# بیش بینی y با شیب خط محاسبه شده
plt.plot(x_values, [(slop_of_primary_values * x) + numbers_mean(b_values, 1) for x in x_values],
         color='yellow')

# بیش بینی y با Numpy
plt.plot(x_data, slope * x_data + intercept, color='green')


plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Graph of the equation y = ax + b')

# تنظیم نام‌گذاری محور x به هر 10 واحد
x_ticks = range(0, 101, 10)  # از 0 تا 100 با گام 10
plt.xticks(x_ticks)

# نمایش نمودار
plt.show()
