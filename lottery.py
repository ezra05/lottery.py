import requests
from bs4 import BeautifulSoup
import csv
import os
import pandas as pd
import logging
from itertools import combinations
import random

logging.basicConfig(level=logging.INFO)

try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    LIGHTGBM_AVAILABLE = True
except ModuleNotFoundError:
    LIGHTGBM_AVAILABLE = False
    logging.error("LightGBM no está instalado. Las funciones de ML no estarán disponibles.")

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
        try:
            html = fetch_results_static(url)
            results = parse_lottery_results_generic(html)
            save_to_csv(name, results)
        except Exception as e:
            logging.error(f"Error en {name}: {e}")

def analyze_csv(name):
    filename = f"{name}_results.csv"
    if not os.path.isfile(filename):
        logging.error(f"No existe el archivo {filename}. Descarga primero los resultados.")
        return None
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

def train_lgb_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    params = {
        'objective': 'multiclass',
        'num_class': y.shape[1],
        'metric': 'multi_logloss',
        'verbose': -1
    }
    model = lgb.train(params, train_data, valid_sets=[test_data], early_stopping_rounds=10)
    y_pred = model.predict(X_test)
    y_pred_labels = (y_pred > 0.5).astype(int)
    accuracy = (y_pred_labels == y_test.values).mean().mean()
    logging.info(f"LightGBM accuracy: {accuracy:.2f}")
    return model

if __name__ == "__main__":
    update_all_lotteries()

    freq = analyze_csv("GanaMas")
    if freq is not None:
        combos_random = generate_random_combinations(36, 5, freq, count=5)
        logging.info(f"Combinaciones aleatorias sugeridas para GanaMas:\n{combos_random}")

        if LIGHTGBM_AVAILABLE:
            csv_file = "GanaMas_results.csv"
            X, y = prepare_ml_data(csv_file)
            model = train_lgb_model(X, y)
        else:
            logging.warning("No se puede entrenar el modelo ML porque LightGBM no está disponible.")
