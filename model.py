#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Игнорирование предупреждений
import warnings
warnings.filterwarnings("ignore")

# Импорт необходимых библиотек:
import numpy as np
import pandas as pd
from matplotlib import pyplot
from re import search

np.random.seed(42)

# Считывание тренировочных данных
df = pd.read_csv('train_dataset_train.csv', sep=',', header=0,
                 parse_dates= ["Дата_Рождения"], index_col=None)

# Вывод сведений о наборе данных:
df.info()

# Заменяем NaN нулями
df.replace(np.nan, 0, inplace=True)

# Описание имеющихся признаков
description = df.describe()

# Сведения о целевой переменной
df['Статус'].value_counts()

###############################################################################
# Обработка "Год_Поступления"
# Ящик с усами
df['Год_Поступления'].plot(kind='box')
pyplot.show()

# df['Год_Поступления'].value_counts()
df=df[df['Год_Поступления'] < 2022]

###############################################################################
# Обработка "СрБаллАттестата"
# Ящик с усами
df['СрБаллАттестата'].plot(kind='box')
pyplot.show()

df['СрБаллАттестата'].value_counts()

# Перебор признака с учётом максимального кол-ва баллов по 3-м предметам
def certificate(column):
    lst = []
    for item in column:
        if item <= 300:
            lst.append(item)
            pass
        else:
            lst.append(-1)
            pass
        pass
    return pd.DataFrame(lst)

df['СрБаллАттестата'] = certificate(df['СрБаллАттестата'])

# Перебор признака для определения доп. столбца - аттестатов ООО и СОО
def add_kind_certificate(column):
    lst = []
    for item in column:
        if item <= 5:
            lst.append("Аттестат_ООО")
            pass
        else:
            lst.append("Аттестат_СОО")
            pass
        pass
    return pd.DataFrame(lst)

df['СрБалл_Вид_Аттестата'] = add_kind_certificate(df['СрБаллАттестата'].astype('float64'))

df['СрБалл_Вид_Аттестата'].value_counts()

###############################################################################
# Обработка "Дата_Рождения"
# Разбивка признака на отдельные столбцы - год, месяц и день - без учёта времени
df['Дата_Рождения_Год'] = df['Дата_Рождения'].dt.year
df['Дата_Рождения_Месяц'] = df['Дата_Рождения'].dt.month
df['Дата_Рождения_День'] = df['Дата_Рождения'].dt.day
df = df.drop(columns=['Дата_Рождения'])

# Вывод формы df
print(df.shape)

# Проверка первых 5-ти строк
print(df[['Дата_Рождения_Год', 'Дата_Рождения_Месяц', 'Дата_Рождения_День']].head(5))

###############################################################################
# Обработка "Пособие"
df['Пособие'].value_counts()
df = df.drop(columns=['Пособие'])

# Вывод формы df
print(df.shape)

###############################################################################
# Обработка "Изучаемый_Язык"
df['Изучаемый_Язык'].value_counts()

def replace_language(column):
    lst = []
    for item in column:
        if item == 0:
            lst.append('Неизвестно')
            pass
        elif search("англи", item.lower()):
            lst.append("Английский")
            pass
        elif search("немецкий", item.lower()):
            lst.append("Немецкий")
            pass
        elif search("французский", item.lower()):
            lst.append("Французский")
            pass
        elif search("русский", item.lower()):
            lst.append("Русский")
            pass
        else:
            lst.append(item)
            pass
        pass
    return pd.DataFrame(lst)

df['Изучаемый_Язык'] = replace_language(df['Изучаемый_Язык'])

df['Изучаемый_Язык'].value_counts()

###############################################################################
# Обработка "Страна_ПП"
df['Страна_ПП'].value_counts()

def replace_country(column):
    lst = []
    for item in column:
        if item == 0:
            lst.append('Неизвестно')
            pass
        elif search("рос", item.lower()):
            lst.append("Россия")
            pass
        elif search("казах", item.lower()):
            lst.append("Казахстан")
            pass
        elif search("тадж", item.lower()):
            lst.append("Таджикистан")
            pass
        elif search("кир", item.lower()):
            lst.append("Киргизия")
            pass
        elif search("кыр", item.lower()):
            lst.append("Киргизия")
            pass
        elif search("китай", item.lower()):
            lst.append("Китай")
            pass
        else:
            lst.append("Другое")
            pass
        pass
    return pd.DataFrame(lst)

df['Страна_ПП'] = replace_country(df['Страна_ПП'])

df['Страна_ПП'].value_counts()

###############################################################################
# Обработка "Регион_ПП"
df['Регион_ПП'].value_counts()

# Список с уникальными значениями
region_lst = df['Регион_ПП'].unique()

def replace_region(column):
    lst = []
    for item in column:
        if item == 0:
            lst.append('Неизвестно')
            pass
        elif search("алтай", item.lower()):
            if search("респ", item.lower()):
                lst.append("Республика Алтай")
                pass
            else:
                lst.append("Алтайский край")
                pass
            pass
        elif search("жалал", item.lower()):
            lst.append("Джалал-Абадская область")
            pass
        elif search("казах", item.lower()):
            if search("респ", item.lower()):
                lst.append("Республика Казахстан")
                pass
            lst.append("Казахстанская область")
            pass
        elif search("саха", item.lower()):
            if search("респ", item.lower()):
                lst.append("Республика Саха (Якутия)")
                pass
            lst.append("Сахалинская область")
            pass
        elif search("иркутская", item.lower()):
            lst.append("Иркутская область")
            pass
        elif search("ошская", item.lower()):
            lst.append("Ошская область")
            pass
        elif search("кемеровская", item.lower()):
            lst.append("Кемеровская область")
            pass
        elif search("алматинская", item.lower()):
            lst.append("Алматинская область")
            pass
        elif search("москов", item.lower()):
            lst.append("Московская область")
            pass
        elif search("новосиб", item.lower()):
            lst.append("Новосибирская область")
            pass
        elif search("тыва", item.lower()):
            lst.append("Республика тыва")
            pass
        elif search("акмолин", item.lower()):
            lst.append("Акмолинская область")
            pass
        elif search("магадан", item.lower()):
            lst.append("Магаданская область")
            pass
        elif search("пров", item.lower()):
            lst.append("провинции Китая")
            pass
        else:
            lst.append("Другое")
            pass
        pass
    return pd.DataFrame(lst)

df['Регион_ПП'] = replace_region(df['Регион_ПП'])

df['Регион_ПП'].value_counts()

del region_lst

###############################################################################
# Обработка "Город_ПП"
df['Город_ПП'].value_counts()

# Список с уникальными значениями
city_lst = df['Город_ПП'].unique()

def current_city(column):
    lst = []
    for item in column:
        if item == 0:
            lst.append('Неизвестно')
            pass
        elif search("барн", item.lower()):
            lst.append("Барнаул")
            pass
        elif search("бий", item.lower()):
            lst.append("Бийск")
            pass
        elif search("горно-ал", item.lower()):
            lst.append("Горно-Алтайск")
            pass
        elif search("новоалт", item.lower()):
            lst.append("Новоалтайск")
            pass
        elif search("белокуриха", item.lower()):
            lst.append("Белокуриха")
            pass
        else:
            # lst.append(item)
            lst.append("Другое")
            pass
        pass
    return pd.DataFrame(lst)

df['Город_ПП'] = current_city(df['Город_ПП'])

df['Город_ПП'].value_counts()

del city_lst

###############################################################################
# Обработка "Страна_Родители"
df['Страна_Родители'].value_counts()

def parents_country(column):
    lst = []
    for item in column:
        if item == 0:
            lst.append('Неизвестно')
            pass
        elif search("россия", item.lower()):
            lst.append("Россия")
            pass
        elif search("казахстан", item.lower()):
            lst.append("Казахстан")
            pass
        elif search("таджикистан", item.lower()):
            lst.append("Таджикистан")
            pass
        elif search("кыр", item.lower()):
            lst.append("Киргизия")
            pass
        elif search("кнр", item.lower()):
            lst.append("Китай")
        elif search("туркменистан", item.lower()):
            lst.append("Туркменистан")
            pass
        elif search("киргизия", item.lower()):
            lst.append("Киргизия")
            pass
        elif search("китай", item.lower()):
            lst.append("Китай")
            pass
        else:
            lst.append("Другое")
            pass
        pass
    return pd.DataFrame(lst)

df['Страна_Родители'] = parents_country(df['Страна_Родители'])

df['Страна_Родители'].value_counts()

###############################################################################
# Обработка "Пол"
df['Пол'].value_counts()

def gender(column):
    lst = []
    for item in column:
        if item == 0:
            lst.append('Не_определился')
            pass
        elif search("муж", item.lower()):
            lst.append("Муж")
            pass
        else:
            lst.append(item)
        pass
    return pd.DataFrame(lst)

df['Пол'] = gender(df['Пол'])

df['Пол'].value_counts()

###############################################################################
# Обработка "Уч_Заведение"
df['Уч_Заведение'].value_counts()

# Список с уникальными значениями
ed_institution_lst = df['Уч_Заведение'].unique()

def ed_institution(column):
    lst = []
    for item in column:
        if item == 0:
            lst.append('Неизвестно')
            pass
        elif search("школ", item.lower()):
            lst.append("Школа")
            pass
        elif search("сош", item.lower()):
            lst.append("Школа")
            pass
        elif search("КГУ", item):
            lst.append("Школа")
            pass
        elif search("ОШ", item):
            lst.append("Школа")
            pass
        elif search("СШ", item):
            lst.append("Школа")
            pass
        elif search("инстит", item.lower()):
            lst.append("Институт")
            pass
        elif search("универ", item.lower()):
            lst.append("Университет")
            pass
        elif search("коллед", item.lower()):
            lst.append("Колледж")
            pass
        elif search("лице", item.lower()):
            lst.append("Лицей")
            pass
        elif search("гимназ", item.lower()):
            lst.append("Гимназия")
            pass
        elif search("академ", item.lower()):
            lst.append("Академия")
            pass
        elif search("техник", item.lower()):
            lst.append("Техникум")
            pass
        elif search("учил", item.lower()):
            lst.append("Училище")
            pass
        elif search("оу впо", item.lower()):
            lst.append("Вуз")
            pass
        else:
            lst.append("Другое")
        pass
    return pd.DataFrame(lst)

df['Уч_Заведение'] = ed_institution(df['Уч_Заведение'])

df['Уч_Заведение'].value_counts()

del ed_institution_lst

###############################################################################
# Обработка "Где_Находится_УЗ"
df['Где_Находится_УЗ'].value_counts()

# Список с уникальными значениями
location_lst = df['Где_Находится_УЗ'].unique()

def location(column):
    lst = []
    for item in column:
        if item == 0:
            lst.append('Неизвестно')
            pass
        elif search("барн", item.lower()):
            lst.append("Барнаул")
            pass
        elif search("бий", item.lower()):
            lst.append("Бийск")
            pass
        elif search("горно-ал", item.lower()):
            lst.append("Горно-Алтайск")
            pass
        elif search("новоалт", item.lower()):
            lst.append("Новоалтайск")
            pass
        elif search("белокуриха", item.lower()):
            lst.append("Белокуриха")
            pass
        elif search("москва", item.lower()):
            lst.append("Москва")
            pass
        elif search("рубцов", item.lower()):
            lst.append("Рубцовск")
            pass
        else:
            lst.append("Другое")
            pass
        pass
    return pd.DataFrame(lst)

df['Где_Находится_УЗ'] = location(df['Где_Находится_УЗ'])

df['Где_Находится_УЗ'].value_counts()

del location_lst

###############################################################################
# Обработка "Опекунство"
df['Опекунство'].value_counts()
df = df.drop(columns=['Опекунство'])

# Вывод формы df
print(df.shape)

# Анализирование оставшихся признаков
df['Основания'].value_counts()
# Не категориальные признаки
df['КодФакультета'].value_counts()
df['Село'].value_counts()
df['Иностранец'].value_counts()
df['Общежитие'].value_counts()
df['Наличие_Матери'].value_counts()
df['Наличие_Отца'].value_counts()
df['Год_Окончания_УЗ'].value_counts()

###############################################################################
# Проверка на наличие nan
notna = df.notna()
tail = notna.tail(5)
df.info() # Если len(column) < 13583, то dropna()
# Удаление последней строки с nan
df.dropna(inplace=True)
del notna, tail

del description

###############################################################################
# Унитарное кодирование (One Hot Encoding) части признаков
df = pd.get_dummies(df, columns=['Пол', 'Основания', 'Изучаемый_Язык','СрБалл_Вид_Аттестата'],
                    prefix=['Пол', 'Основание', 'Изуч.язык', 'СрБалл_Аттестата'])

head = df.head(5)
del head

###############################################################################
# Вывод распределения признаков по типу
df.dtypes

# Кодирование части признаков с помощью метки (Label Encoding)
df['Страна_ПП_кодир'] = df['Страна_ПП'].astype('category').cat.codes
df['Регион_ПП_кодир'] = df['Регион_ПП'].astype('category').cat.codes
df['Город_ПП_кодир'] = df['Город_ПП'].astype('category').cat.codes
df['Страна_Родители_кодир'] = df['Страна_Родители'].astype('category').cat.codes
df['Уч_Заведение_кодир'] = df['Уч_Заведение'].astype('category').cat.codes
df['Где_Находится_УЗ_кодир'] = df['Где_Находится_УЗ'].astype('category').cat.codes

# Удаление исходных признаков после кодирования
df = df.drop(columns=['Уч_Заведение', 'Где_Находится_УЗ', 'Страна_ПП',
                      'Регион_ПП', 'Город_ПП', 'Страна_Родители'])

# Проверка выполнения
head = df.head(5)
tail = df.tail(5)

del head, tail

###############################################################################

# Матрица коэффициентов корреляции по Пирсону
correlations = df.corr(method='pearson')
# Корреляция с выходной (целевой) переменной
cor_target = abs(correlations['Статус'])

###############################################################################
# Выполнение Oversampling'a миноритарного класса
df['Статус'].value_counts()

df_1 = df.loc[df['Статус']==-1]
df = pd.concat([df, df_1]).sample(frac=1)
del df_1

###############################################################################
###############################################################################
# Построение модели
# Выделение зависимой (целевой) переменной y
y = df['Статус'].values
# Выделение независимых переменных-предикторов X
X = df.drop(['Статус'], axis=1).values

from xgboost import XGBClassifier
model =  XGBClassifier()
model.fit(X, y)

# Фиксация порядка расположения признаков и их количества
colls = df.drop(['Статус'], axis=1).columns.values

###############################################################################
###############################################################################
# Загрузка тестовой части набора данных
df_test = pd.read_csv('test_dataset_test.csv', sep=',',
                      header=0, parse_dates= ["Дата_Рождения"], index_col=None)

# Заменяем NaN нулями
df_test.replace(np.nan, 0, inplace=True)

# Удаление неинформативных признаков
df_test = df_test.drop(columns=['Пособие'])
df_test = df_test.drop(columns=['Опекунство'])

# Обработка признаков
df_test['СрБаллАттестата'] = certificate(df_test['СрБаллАттестата'].astype('float64'))
df_test['Дата_Рождения_Год'] = df_test['Дата_Рождения'].dt.year
df_test['Дата_Рождения_Месяц'] = df_test['Дата_Рождения'].dt.month
df_test['Дата_Рождения_День'] = df_test['Дата_Рождения'].dt.day
df_test = df_test.drop(columns=['Дата_Рождения'])
df_test['СрБалл_Вид_Аттестата'] = add_kind_certificate(df_test['СрБаллАттестата'].astype('float64'))
df_test['Изучаемый_Язык'] = replace_language(df_test['Изучаемый_Язык'])
df_test['Страна_ПП'] = replace_country(df_test['Страна_ПП'])
df_test['Регион_ПП'] = replace_region(df_test['Регион_ПП'])
df_test['Город_ПП'] = current_city(df_test['Город_ПП'])
df_test['Страна_Родители'] = parents_country(df_test['Страна_Родители'])
df_test['Пол'] = gender(df_test['Пол'])
df_test['Уч_Заведение'] = ed_institution(df_test['Уч_Заведение'])
df_test['Где_Находится_УЗ'] = location(df_test['Где_Находится_УЗ'])

# Кодирование и удаление исходных признаков после кодирования
df_test = pd.get_dummies(df_test, columns=['Пол', 'Основания', 'Изучаемый_Язык', 'СрБалл_Вид_Аттестата'],
                    prefix=['Пол', 'Основание', 'Изуч.язык', 'СрБалл_Аттестата'])
df_test['Страна_ПП_кодир'] = df_test['Страна_ПП'].astype('category').cat.codes
df_test['Регион_ПП_кодир'] = df_test['Регион_ПП'].astype('category').cat.codes
df_test['Город_ПП_кодир'] = df_test['Город_ПП'].astype('category').cat.codes
df_test['Страна_Родители_кодир'] = df_test['Страна_Родители'].astype('category').cat.codes
df_test['Уч_Заведение_кодир'] = df_test['Уч_Заведение'].astype('category').cat.codes
df_test['Где_Находится_УЗ_кодир'] = df_test['Где_Находится_УЗ'].astype('category').cat.codes

df_test = df_test.drop(columns=['Уч_Заведение', 'Где_Находится_УЗ', 'Страна_ПП',
                      'Регион_ПП', 'Город_ПП', 'Страна_Родители'])

# Упорядочивание признаков
df_test = df_test[colls]

# Формирование независимых переменных-предикторов X_test
X_test = df_test

# Выполнение предсказаний
yhat_test = model.predict(X_test)

# Сохранение результатов для отправки на проверку
yhat_test = pd.DataFrame(yhat_test, columns=['Статус'])
df_test_sample = pd.read_csv('sample_submission.csv', sep=',', header=0)
df_test_sample['Статус'] = yhat_test
df_test_sample.to_csv('sample_submission.csv', index=False)
