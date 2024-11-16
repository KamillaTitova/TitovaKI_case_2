import pandas as pd
from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)
# Загружаем модель и определяем параметры функции  -  будущие входы для модели (всего 12 параметров)

def set_params(params):
    pickle_model_file = './Models/dtr_d_model.pkl'
    pickle_scaler_x_file = './Models/scaler_x.pkl'
    with open(pickle_model_file, 'rb') as file:
        pickle_model = pickle.load(file)
    with open(pickle_scaler_x_file, 'rb') as file:
        norma_x = pickle.load(file)
    x = np.array(params).reshape(1, -1)
    params_norma = norma_x.transform(np.array(params).reshape(1, -1))
    y_predict = pickle_model.predict(params_norma)
    #y_predict = pickle_model.predict(x)
    return y_predict[0]


@app.route('/', methods=['post', 'get'])
def processing():
    input_params = []
    message = ''
    target_variable = 'Прочность при растяжении, МПа'
    if request.method == 'POST':
        # получим данные из наших форм и кладем их в список, который затем передадим функции set_params
        columns = ["Соотношение матрица-наполнитель",
                   "Плотность, кг/м3",
                   "Модуль упругости, ГПа",
                   "Количество отвердителя, м.%",
                   "Содержание эпоксидных групп,%_2",
                   "Температура вспышки, С_2",
                   "Поверхностная плотность, г/м2",
                   "Потребление смолы, г/м2",
                   "Потребление смолы, г/м2",
                   "Угол нашивки, град",
                   "Шаг нашивки",
                   "Плотность нашивки"]

        for i in range(1, 13):
            input_params.append(request.form.get(f'param{i}', type=float))
        print(input_params)
        predicted_value = set_params(input_params)
        print(predicted_value)
        message = f"{target_variable}: {predicted_value}"
    return render_template('index.html', message=message)


if __name__ == '__main__':
    app.run()