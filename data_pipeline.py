"""Carregamento e preparação do dataset brasileiro de COVID-19."""

from pathlib import Path
import numpy as np
import pandas as pd
import requests

DATA_URL = "https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv"
DATA_DIR = Path(__file__).parent / "data"
CACHE_FILE = DATA_DIR / "cases-brazil-states.csv"


def load_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if not CACHE_FILE.exists():
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        CACHE_FILE.write_bytes(response.content)
    frame = pd.read_csv(CACHE_FILE)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["state"].notna()].copy()
    numeric_columns = ["newCases", "newDeaths", "totalCases", "deaths", "recovered"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    # A fonte atual não publica activeCases; esta é a definição equivalente.
    frame["activeCases"] = (frame["totalCases"] - frame["deaths"] - frame["recovered"]).clip(lower=0)
    frame = frame.sort_values(["state", "date"])
    # Variáveis de entrada usam o dia anterior; o alvo é o crescimento alto de óbitos.
    frame["previous_cases"] = frame.groupby("state")["newCases"].shift(1)
    frame["previous_deaths"] = frame.groupby("state")["newDeaths"].shift(1)
    frame["previous_active"] = frame.groupby("state")["activeCases"].shift(1)
    frame["day_number"] = (frame["date"] - frame["date"].min()).dt.days
    frame = frame.dropna(subset=["previous_cases", "previous_deaths", "previous_active", "newDeaths"])
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    threshold = frame["newDeaths"].quantile(0.75)
    frame["target"] = (frame["newDeaths"] >= threshold).astype(int)
    features = ["previous_cases", "previous_deaths", "previous_active", "day_number"]
    x = frame[features].to_numpy(dtype=float)
    y = frame[["target"]].to_numpy(dtype=float)
    # Divisão temporal para evitar vazamento: os últimos 20% são o teste.
    split = int(len(frame) * 0.8)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0] = 1
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test, y_train, y_test, frame.iloc[split:].copy(), float(threshold)
