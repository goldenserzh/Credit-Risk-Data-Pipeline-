import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns

def load_data(train_path: str, bureau_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Загружает данные из CSV-файлов. """
    app_train = pd.read_csv(train_path)
    bureau = pd.read_csv(bureau_path)
    app_test = pd.read_csv(test_path)
    return app_train, bureau, app_test

def calculate_total_debt(app_train: pd.DataFrame, app_test: pd.DataFrame, bureau: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Рассчитывает общий долг заемщика. """
    bureau_debt = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum().reset_index()
    bureau_debt.rename(columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_DEBT'}, inplace=True)

    app_train = app_train.merge(bureau_debt, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(bureau_debt, on='SK_ID_CURR', how='left')

    app_train['TOTAL_DEBT'].fillna(0, inplace=True)
    app_test['TOTAL_DEBT'].fillna(0, inplace=True)

    return app_train, app_test

def calculate_dti(app_train: pd.DataFrame, app_test: pd.DataFrame):
     # Убедись, что возвращаешь DataFrame
    app_train['DTI'] = (app_train['AMT_ANNUITY'] + app_train['TOTAL_DEBT']) / app_train['AMT_INCOME_TOTAL']
    app_test['DTI'] = (app_test['AMT_ANNUITY'] + app_test['TOTAL_DEBT']) / app_test['AMT_INCOME_TOTAL']
    return app_train, app_test


def calculate_previous_credits(app_train: pd.DataFrame, app_test: pd.DataFrame, bureau: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Подсчитывает количество прошлых кредитов заемщика. """
    prev_credit = bureau.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count().reset_index()
    prev_credit.rename(columns={'SK_ID_BUREAU': 'AMOUNT_CREDITS'}, inplace=True)

    app_train = app_train.merge(prev_credit, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(prev_credit, on='SK_ID_CURR', how='left')

    app_train['AMOUNT_CREDITS'].fillna(0, inplace=True)
    app_test['AMOUNT_CREDITS'].fillna(0, inplace=True)

    return app_train, app_test


def analyze_credit_history(app_train: pd.DataFrame) -> pd.Series:
    """ Анализирует среднее количество прошлых кредитов. """
    return app_train.groupby('TARGET')['AMOUNT_CREDITS'].mean()

def plot_dti_distribution(app_train: pd.DataFrame) -> None:
    """
    Строит гистограмму распределения DTI (Debt-to-Income Ratio) 
    для заемщиков с дефолтом и без.

    :param app_train: DataFrame с заявками на кредит, содержащий колонки 'DTI' и 'TARGET'.
    """
    plt.figure(figsize=(15, 8))
    
    # Гистограмма для заемщиков без дефолта
    sns.histplot(app_train[app_train['TARGET'] == 0]['DTI'], 
                 bins=50, color='green', label='No Default', kde=True)
    
    # Гистограмма для заемщиков с дефолтом
    sns.histplot(app_train[app_train['TARGET'] == 1]['DTI'], 
                 bins=50, color='red', label='Default', kde=True)
    
    # Оформление графика
    plt.xlabel('DTI')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Распределение коэффициента долговой нагрузки (DTI)')
    plt.xlim(0, 100)
    plt.show()


def plot_credit_amount_by_target(app_train: pd.DataFrame) -> None:
    """
    Строит boxplot (ящик с усами) для количества кредитов заемщиков 
    в зависимости от статуса дефолта.

    :param app_train: DataFrame с заявками на кредит, содержащий колонки 'TARGET' и 'AMOUNT_CREDITS'.
    """
    plt.figure(figsize=(12, 9))
    
    # Построение boxplot
    sns.boxplot(x=app_train['TARGET'], y=app_train['AMOUNT_CREDITS'])
    
    # Настройка подписей
    plt.xticks([0, 1], ['No Default', 'Default'])
    plt.xlabel('Статус заемщика')
    plt.ylabel('Число кредитов')
    plt.title('Распределение числа кредитов в зависимости от дефолта')
    
    # Отображение графика
    plt.show()


def plot_credit_amount_by_target(app_train: pd.DataFrame) -> None:
    """
    Строит boxplot (ящик с усами) для количества кредитов заемщиков 
    в зависимости от статуса дефолта.

    :param app_train: DataFrame с заявками на кредит, содержащий колонки 'TARGET' и 'AMOUNT_CREDITS'.
    """
    plt.figure(figsize=(12, 9))
    
    # Построение boxplot
    sns.boxplot(x=app_train['TARGET'], y=app_train['AMOUNT_CREDITS'])
    
    # Настройка подписей
    plt.xticks([0, 1], ['No Default', 'Default'])
    plt.xlabel('Статус заемщика')
    plt.ylabel('Число кредитов')
    plt.title('Распределение числа кредитов в зависимости от дефолта')
    
    # Отображение графика
    plt.show()

def plot_age_distribution(app_train: pd.DataFrame, app_test: pd.DataFrame) -> None:
    """
    Строит гистограмму распределения возраста заемщиков для 
    дефолтных и недефолтных клиентов.

    :param app_train: DataFrame с заявками на кредит, содержащий колонки 'DAYS_BIRTH' и 'TARGET'.
    """
    # Преобразование возраста из дней в годы
    app_train['AGE'] = -(app_train['DAYS_BIRTH'] // 365)
    app_test['AGE'] = -(app_train['DAYS_BIRTH'] // 365)
    
    app_train = app_train.drop('DAYS_BIRTH', axis=1)
    app_test = app_test.drop('DAYS_BIRTH', axis=1)

    plt.figure(figsize=(12, 8))
    
    # Гистограмма для недефолтных заемщиков
    sns.histplot(app_train[app_train['TARGET'] == 0]['AGE'], 
                 bins=25, color='green', kde=True, label='Недефолтные')

    # Гистограмма для дефолтных заемщиков
    sns.histplot(app_train[app_train['TARGET'] == 1]['AGE'], 
                 bins=25, color='red', kde=True, label='Дефолтные')

    # Оформление графика
    plt.xlabel('Возраст')
    plt.ylabel('Число заемщиков')
    plt.legend()
    plt.title('Распределение возраста заемщиков')

    # Отображение графика
    plt.show()

def divisionByAge(age= int) -> int:
    if age < 25:
        return 0
    elif age < 50:
        return 1
    else: return 2

def plot_income_distribution(app_train: pd.DataFrame) -> None:
    """
    Строит гистограмму распределения доходов заемщиков для 
    дефолтных и недефолтных клиентов.

    :param app_train: DataFrame с заявками на кредит, содержащий колонки 'AMT_INCOME_TOTAL' и 'TARGET'.
    """
    # Вычисление среднего дохода для дефолтных и недефолтных заемщиков
    salary_0 = app_train[app_train['TARGET'] == 0]['AMT_INCOME_TOTAL'].mean()
    salary_1 = app_train[app_train['TARGET'] == 1]['AMT_INCOME_TOTAL'].mean()

    plt.figure(figsize=(18, 8))
    
    # Гистограмма для недефолтных заемщиков
    sns.histplot(app_train[app_train['TARGET'] == 0]['AMT_INCOME_TOTAL'], 
                 bins=10, color='green', kde=True, label='Недефолтные')

    # Гистограмма для дефолтных заемщиков
    sns.histplot(app_train[app_train['TARGET'] == 1]['AMT_INCOME_TOTAL'], 
                 bins=50, color='red', kde=True, label='Дефолтные')

    # Оформление графика
    plt.xlabel('Доход заемщиков')
    plt.ylabel('Количество заемщиков')
    plt.title('Распределение доходов заемщиков')
    plt.legend()
    plt.xlim(0, 6_000_000)
    plt.ylim(0, 500_000)
    
    # Отображение графика
    plt.show()


def plot_income_boxplot(app_train: pd.DataFrame) -> None:
    """
    Строит boxplot (ящик с усами) для доходов заемщиков 
    в зависимости от статуса дефолта.

    :param app_train: DataFrame с заявками на кредит, содержащий колонки 'AMT_INCOME_TOTAL' и 'TARGET'.
    """
    plt.figure(figsize=(10, 10))
    
    # Построение boxplot
    sns.boxplot(x=app_train['TARGET'], y=app_train['AMT_INCOME_TOTAL'])
    
    # Настройка подписей
    plt.xticks(ticks=[0, 1], labels=['No Default', 'Default'])
    plt.xlabel('Статус заемщика')
    plt.ylabel('Доход заемщика')
    plt.title('Распределение доходов заемщиков по статусу дефолта')
    
    # Ограничение по оси Y для лучшей визуализации
    plt.ylim(0, 1_000_000)
    
    # Отображение графика
    plt.show()


def child_classification(amount: int) -> str:
    if amount == 0:
        return "childless family"
    elif amount >= 1 and amount  <= 2:
        return 'An ordinary family' 
    elif 3 <= amount <= 8:
        return "Big family"
    else:
        return 'Huge family'
    
def plot_family_category_distribution(app_train: pd.DataFrame) -> None:
    """
    Строит гистограмму распределения типов семей заемщиков 
    для дефолтных и недефолтных клиентов.

    :param app_train: DataFrame с заявками на кредит, содержащий колонки 'FAMILY_CATEGORY' и 'TARGET'.
    """
    plt.figure(figsize=(12, 8))
    
    # Гистограмма для недефолтных заемщиков
    sns.histplot(app_train[app_train['TARGET'] == 0]['FAMILY_CATEGORY'], 
                 color='green', kde=False, label='Недефолтные')

    # Гистограмма для дефолтных заемщиков
    sns.histplot(app_train[app_train['TARGET'] == 1]['FAMILY_CATEGORY'], 
                 color='red', kde=False, label='Дефолтные')

    # Оформление графика
    plt.xlabel('Тип семьи')
    plt.ylabel('Количество заемщиков')
    plt.title('Распределение типов семей заемщиков по статусу дефолта')
    plt.legend()
    plt.ylim(0, 500_000)

    # Отображение графика
    plt.show()