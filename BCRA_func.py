import pandas as pd
import requests
import json
import datetime as dt


def errorback(message):
    print(message)


def json_df(url, header):
    try:
        response_API = requests.get(url, headers=header)
        data = response_API.text
        parse_json = json.loads(data)
        return pd.DataFrame(parse_json)
    except Exception as e:
        errorback(e)


def insert_df(data1, data2, col, on):
    try:
        df = pd.concat([data1[f'{col}'], data2], axis=1)
        return data1.merge(df, on=on)
    except Exception as e:
        errorback(e)


def calc_vol(data, fro_m):
    """

    """
    try:
        data['PCT %'] = data[f'{fro_m}'].pct_change()
        data_pct = data['PCT %'].iloc[1:]
        vol = data_pct.rolling(20).var(ddof=1)
        vol.name = 'Vol'
        return vol
    except Exception as e:
        errorback(e)


def calc_week(data, on):
    try:
        weeks = data[f'{on}'].dt.isocalendar()
        weeks.drop(columns={'year', 'day'}, inplace=True)
        weeks = insert_df(data, weeks, on, on)
        return weeks
    except Exception as e:
        errorback(e)


def re_idx_name(data, ogcol, rename=False):
    try:
        data = data.reindex(ogcol, axis=1)
        if rename:
            data.rename(columns=rename, inplace=True)
            return data
        return data
    except TypeError as e:
        print('TypeError: Insert a dict in name place')
    except Exception as e:
        return e


def clean_reg(data, i_date, f_date, on, drop_cols):
    try:
        date_int = data.loc[(data[f'{on}'] > i_date) & (data[f'{on}'] < f_date)]
        reg_data = date_int.drop(columns=drop_cols)
        reg_data[f'{on}'] = reg_data[f'{on}'].map(dt.datetime.toordinal)
        return reg_data
    except TypeError:
        print('TypeError: Insert a dict in name place')
    except Exception as e:
        return e


def reg_prep_data(data1, data2):
    try:
        x = data1.values
        y = data2.values
        x = x.reshape(-1, 1)
        return x, y
    except Exception as e:
        return e


def pred_usd(fecha):
    try:
        pred_date = dt.datetime(fecha[0], fecha[1], fecha[2])
        pred_dateord = pred_date.toordinal()
        print('$', round(model.predict([[pred_dateord]])[0], 2))
    except Exception as e:
        return e


def main():
    pass

if __name__ == '__main__':
    main()
