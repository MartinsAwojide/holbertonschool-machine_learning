#!/usr/bin/env python3
"""Preprocess dataset"""
import pandas as pd


def preprocess():
    add1 = "data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"
    add2 = "data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
    df_bit = pd.read_csv(add1)
    df_coin = pd.read_csv(add2)
    df_coin['Timestamp'] = pd.to_datetime(df_coin['Timestamp'], unit='s')
    df_bit['Timestamp'] = pd.to_datetime(df_bit['Timestamp'], unit='s')
    #new_df = pd.concat([df_bit, df_coin], ignore_index=True)
    new_df = df_coin[df_coin['Timestamp'] >= '2017'].copy()
    #new_df.drop_duplicates(subset="Timestamp", inplace=True)
    new_df.index = new_df['Timestamp']
    new_df[new_df.columns.values] = new_df[new_df.columns.values].ffill()
    new_df = new_df.resample('H').agg({'Open': 'first', 'High': 'max', 'Low': 'min',
                                       'Close': 'last', 'Volume_(BTC)': 'sum',
                                       'Volume_(Currency)': 'sum', 'Weighted_Price': 'mean'})
    new_df.ffill(inplace=True)
    new_df.to_csv("bitcoin.csv")
    return new_df
