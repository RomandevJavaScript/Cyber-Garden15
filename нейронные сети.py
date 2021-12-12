import numpy as np
from matplotlib import pyplot as plt
import filterpy
import keras
import pillow 


def init_net():
    input_nodes = 784
    print('Ввести число скрытых нейронов: ')
    hidden_nodes = int(input())
    out_nodes = 10
    print('Ввести скорость обучения(0.5): ')
    lern_node = float(input())
    return input_nodes, hidden_nodes, out_nodes, lern_node
    import numpy

def creat_net(input_nodes, hidden_nodes, out_nodes,):
    # сознание массивов. -0.5 вычитаем что бы получить диапазон -0.5 +0.5 для весов
    input_hidden_w = (numpy.random.rand(hidden_nodes, input_nodes) - 0.5)
    hidden_out_w = (numpy.random.rand(out_nodes, hidden_nodes) - 0.5)
    return input_hidden_w, hidden_out_w

import scipy.special 
def fun_active(x):
    return scipy.special.expit(x)

def treyn(targget_list,input_list, input_hidden_w, hidden_out_w, lern_node):
    #Прогоняем данные через сеть
    targgets = numpy.array(targget_list, ndmin=2).T
    inputs_sig = numpy.array(input_list, ndmin=2).T
    hidden_inputs = numpy.dot(input_hidden_w, inputs_sig)
    hidden_out = fun_active(hidden_inputs)
    final_inputs = numpy.dot(hidden_out_w, hidden_out)
    final_out = fun_active(final_inputs)
    #Рассчитываем ошибку выходного слоя
    out_errors = targgets - final_out
    #Рассчитываем ошибку скрытого слоя
    hidden_errors = numpy.dot(hidden_out_w.T, out_errors)


hidden_out_w += lern_node * numpy.dot((out_errors * final_out*(1 - final_out)), numpy.transpose(hidden_out))
input_hidden_w += lern_node * numpy.dot((hidden_errors * hidden_out*(1-hidden_out)), numpy.transpose(inputs_sig))



# импорт pandas
import pandas as pd
# Считайте DataFrame, используя данные функции
df = pd.DataFrame(data.data, columns=data.feature_names)
# Добавьте столбец "target" и заполните его данными.
df['target'] = data.target
# Посмотрим первые пять строк
df.head()

def median_blurring():
    image = cv2.imread('girl.jpg')
    img_blur_3 = cv2.medianBlur(image, 3)
    img_blur_7 = cv2.medianBlur(image, 7)
    img_blur_11 = cv2.medianBlur(image, 11)

   print("opencv addition: {}".format(cv2.add(np.uint8([250]), 
                                                   np.uint8([30]))))
print("opencv subtract: {}".format(cv2.subtract(np.uint8([70]), 
                                                    np.uint8([100]))))
print("numpy addition: {}".format(np.uint8([250]) + np.uint8([30])))
print("numpy subtract: {}".format(np.uint8([70]) - np.uint8([71])))

image = cv2.imread('rectangles.png')
b, g, r = cv2.split(image)
cv2.imshow('blue', b)
cv2.imshow('green', g)
cv2.imshow('red', r) 
