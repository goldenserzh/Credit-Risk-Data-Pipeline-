import pandas as pd
from typing import Tuple
def numcat_columns(df: pd.DataFrame):
    df_categor_indexes = df.select_dtypes(['object']).columns.tolist()
    df_numerical_indexes = df.select_dtypes(['int64', 'float64']).columns.tolist()
    return df_categor_indexes, df_numerical_indexes

def fill_missing_categor(df, df_categor_indexes):
    for col in df_categor_indexes:
        df[col] = df[col].fillna(df[col].mode()[0]).astype(str)
    return df

def fill_missing_numeric(df, df_numerical_indexes):
    for col in df_numerical_indexes:
        df[col].fillna(df[col].mean(), inplace = True)

    return df

def calcualte_total_debt(app_train: pd.DataFrame, app_test: pd.DataFrame, bureau:pd.DataFrame):
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

def child_classification(amount: int) -> str:
    if amount == 0:
        return "childless family"
    elif amount >= 1 and amount  <= 2:
        return 'An ordinary family' 
    elif 3 <= amount <= 8:
        return "Big family"
    else:
        return 'Huge family'
    
def family_feature(app_train: pd.DataFrame, app_test: pd.DataFrame):
    app_train['FAMILY_CATEGORY'] = app_train['CNT_CHILDREN'].apply(child_classification)
    app_train.drop('CNT_CHILDREN', axis=1)

    app_test['FAMILY_CATEGORY'] = app_test['CNT_CHILDREN'].apply(child_classification)
    app_test.drop('CNT_CHILDREN', axis=1)
    