import requests
from bs4 import BeautifulSoup
import csv
import os
import pandas as pd
import logging
from itertools import combinations
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)

# URLs oficiales
DOMINICAN_LOTTERIES = {
    "GanaMas": "https://ganamas.com.do/loterias-americanas/resultados",
    "LotoReal": "https://loteriasdominicanas.com/loto-real/resultados"
}

AMERICAN_LOTTERIES = {
    "Powerball": "https://www.powerball.com/es/compruebe-sus-numeros",
    "MegaMillions": "https://www.megamillions.com/Winning-Numbers.aspx"
}

def fetch_results_static(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def parse_lottery_results_generic(html):
    soup = BeautifulSoup(html, "html.parser")
    results = []
    table = soup.find("table")
    if not table:
        logging.warning("No se encontró tabla de resultados")
        return results
    rows = table.find_all("tr")[1:]  # omitimos encabezado
    for row in rows:
        cols = [td.text.strip() for td in row.find_all("td")]
        if len(cols) >= 2:
            result = {"date": cols[0], "numbers": cols[1]}
            if len(cols) > 2:
                result["extra"] = ", ".join(cols[2:])
            else:
                result["extra"] = ""
            results.append(result)
    return results

def save_to_csv(name, results):
    filename = f"{name}_results.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "numbers", "extra"])
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r)
    logging.info(f"{len(results)} resultados guardados en {filename}")

def update_all_lotteries():
    for name, url in {**DOMINICAN_LOTTERIES, **AMERICAN_LOTTERIES}.items():
        logging.info(f"Descargando resultados de {name} ...")
        html = fetch_results_static(url)
        results = parse_lottery_results_generic(html)
        save_to_csv(name, results)

def analyze_csv(name):
    filename = f"{name}_results.csv"
    df = pd.read_csv(filename)
    numbers_series = df["numbers"].str.split(' ').explode()
    freq = numbers_series.value_counts()
    logging.info(f"Frecuencia de números en {name}:\n{freq.head(10)}")
    return freq

def mandel_combinations(n, k):
    return list(combinations(range(1, n+1), k))

def haigh_combinations(n, k, freq):
    most_common = set(int(num) for num in freq.head(5).index)
    less_frequent = [x for x in range(1, n+1) if x not in most_common]
    combos = list(combinations(less_frequent, k))
    return combos

def generate_random_combinations(n, k, freq, count=10):
    combos = []
    most_common = set(int(num) for num in freq.head(5).index)
    pool = [x for x in range(1, n+1) if x not in most_common]
    while len(combos) < count:
        combo = tuple(sorted(random.sample(pool, k)))
        if combo not in combos:
            combos.append(combo)
    return combos

def prepare_ml_data(csv_file):
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').map(pd.Timestamp.toordinal)
    expand_nums = df['numbers'].str.get_dummies(sep=' ')
    X = pd.concat([df['date'], expand_nums], axis=1).fillna(0)
    y = expand_nums
    return X, y

def train_rf_model(X, y):
    X_train, X_test, y_train, y_test
