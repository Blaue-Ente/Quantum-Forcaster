# Adaptive Quantum Financial Forecaster

Уникален хибриден **квантово-класически** генератор на прогнози за финансовите пазари.

Комбинира адаптивен **Variational Quantum Circuit (VQC)** (Qiskit) с класически ML
ансамбъл (Random Forest + Gradient Boosting) за прогнозиране на посоката на
движение на акции, криптовалути и форекс.

## Архитектура

```
quantum-financial-forecaster/
├── backend/
│   ├── __main__.py              # Точка на стартиране
│   ├── core/
│   │   └── config.py            # Конфигурация (квантови + ансамбъл параметри)
│   ├── models/
│   │   ├── data_loader.py       # yfinance + технически индикатори
│   │   ├── quantum_classifier.py # Адаптивен VQC (ZZFeatureMap + RealAmplitudes)
│   │   └── hybrid_forecaster.py  # Хибриден ансамбъл + режим-детекция
│   └── api/
│       └── server.py            # FastAPI сървър
├── frontend/
│   ├── templates/index.html
│   └── static/
│       ├── css/styles.css
│       └── js/app.js            # Дашборд с Chart.js
├── requirements.txt
└── README.md
```

## Какво го прави "уникален и адаптивен"

1. **Адаптивна квантова верига** – броят кубита и дълбочината на ansatz се
   пренастройват според волатилността на актива:
   - при висока волатилност → по-дълбок ansatz (повече експресивност);
   - при ниска волатилност → по-плитък ansatz (по-бързо и стабилно).

2. **Адаптивни тегла на ансамбъла** – според пазарния режим:
   - `volatile` → класическият модел получава ~65% тегло;
   - `trending` → квантовият модел получава повече тегло;
   - `mean-reverting` → балансирани тегла.

3. **Пазарна режим-детекция** – анализира последните 30 дни за волатилност и
   тренд, класифицира режима в `trending` / `mean-reverting` / `volatile`.

4. **Хибридно прогнозиране** – крайната вероятност е претеглена сума от VQC и
   класически ансамбъл; плюс проекция на цена за хоризонта.

## Бърз старт

### 1. Инсталиране на зависимости

```bash
cd quantum-financial-forecaster
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

> **Забележка:** Qiskit и qiskit-machine-learning са големи пакети.
> Инсталацията може да отнеме 2-5 минути.

### 2. Стартиране на сървъра

От папката на проекта (там, където е `backend/`):

```bash
python -m backend
```

или

```bash
uvicorn backend.api.server:app --host 0.0.0.0 --port 8000
```

### 3. Отваряне на дашборда

Браузър: <http://localhost:8000>

1. Изберете категория (акции / крипто / форекс);
2. Изберете актив (напр. `AAPL`, `BTC-USD`, `EURUSD=X`);
3. Задайте хоризонт в дни (1–30);
4. Натиснете **„Генерирай прогноза“**.

Първата прогноза отнема ~30–90 секунди (обучение на квантовата верига).

## API endpoints

| Метод | Път | Описание |
|-------|-----|----------|
| GET | `/` | Уеб дашборд |
| GET | `/api/assets` | Поддържани активи |
| GET | `/api/forecast?symbol=AAPL&horizon=7` | Генерира прогноза |
| GET | `/api/health` | Health check |

Пример:

```bash
curl "http://localhost:8000/api/forecast?symbol=BTC-USD&horizon=7"
```

## Поддържани активи

- **Акции:** AAPL, MSFT, NVDA, TSLA, AMZN
- **Криптовалути:** BTC-USD, ETH-USD, SOL-USD, XRP-USD
- **Форекс:** EURUSD, GBPUSD, USDJPY

Можете да подадете всеки валиден yfinance символ.

## Какво се визуализира

- Историческа цена + проекция (пунктирана линия);
- Схема на квантовата верига (кубити, H/Ry/M гейтове);
- Сравнение P(↑) между VQC, RF, GB и ансамбъла;
- Тегла на ансамбъла (doughnut) според режима;
- Out-of-sample точност на квантовия и класическия модел.

## Технологии

- **Qiskit** + **qiskit-machine-learning** – VQC, ZZFeatureMap, RealAmplitudes
- **scikit-learn** – RandomForest, GradientBoosting
- **yfinance** + **ta** – пазарни данни и технически индикатори
- **FastAPI** + **Uvicorn** – backend
- **Chart.js** – визуализации

## ⚠ Отказ от отговорност

Този проект е само с **образовательна цел**. Не е финансов съвет.
Квантовите симулации се изпълняват на класически симулатор (Aer qasm_simulator)
и са приближения на реални квантови изчисления. Прогнозите не гарантират
бъдещи резултати.
