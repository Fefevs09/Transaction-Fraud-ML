# Transaction Fraud ML

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.5-013243?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Pipeline de Machine Learning para detecção de fraude em transações de cartão de crédito, com engenharia de features, comparação de modelos e ajuste de hiperparâmetros sobre um dataset fortemente desbalanceado (~1% de fraude).

**[Português](#-português)** | **[English](#-english)**

---

## 🇧🇷 Português

### Índice
- [Sobre o projeto](#sobre-o-projeto)
- [Dataset](#dataset)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Tecnologias](#tecnologias)
- [Como executar](#como-executar)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Limitações e próximos passos](#limitações-e-próximos-passos)
- [Licença](#licença)
- [Autor](#autor)

### Sobre o projeto

Este é um projeto de estudo/portfólio que implementa um pipeline completo de detecção de fraude em transações de cartão de crédito: limpeza e engenharia de features, comparação de modelos de classificação (KNN, Regressão Logística e Random Forest) e ajuste de hiperparâmetros via `RandomizedSearchCV`.

Detecção de fraude é um problema clássico de **classificação com classes extremamente desbalanceadas** — a grande maioria das transações é legítima, e é justamente essa característica que torna a escolha das métricas de avaliação tão importante (veja a seção [Resultados](#resultados)).

### Dataset

Os dados seguem o schema do [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection), um conjunto de transações **sintéticas** geradas pelo simulador [Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation), amplamente usado em estudos de detecção de fraude.

| Coluna | Descrição |
|---|---|
| `index` | Identificador único para cada linha |
| `trans_date_trans_time` | Data e hora da transação |
| `cc_num` | Número do cartão de crédito do cliente |
| `merchant` | Nome do comerciante |
| `category` | Categoria do comerciante |
| `amt` | Valor da transação |
| `first` / `last` | Nome / sobrenome do titular do cartão |
| `gender` | Gênero do titular do cartão |
| `street` / `city` / `state` / `zip` | Endereço residencial do titular do cartão |
| `lat` / `long` | Latitude / longitude do titular do cartão |
| `city_pop` | População da cidade do titular do cartão |
| `job` | Profissão do titular do cartão |
| `dob` | Data de nascimento do titular do cartão |
| `trans_num` | Número da transação |
| `unix_time` | Horário UNIX da transação |
| `merch_lat` / `merch_long` | Latitude / longitude do comerciante |
| `is_fraud` | Indicador de fraude — **classe alvo** |

**Desbalanceamento de classes** (medido diretamente em `data/treino.csv`):

| Conjunto | Linhas | Legítimas | Fraude |
|---|---|---|---|
| Treino (`treino.csv`) | 144.353 | 142.666 (≈99,06%) | 1.528 (≈1,06%) |
| Teste (`teste.csv`) | 36.048 | sem rótulo — holdout estilo Kaggle, previsto em `data/submissao.csv` | |

Esse desbalanceamento é o motivo pelo qual acurácia sozinha é uma métrica enganosa aqui: um modelo trivial que sempre prevê "não é fraude" já acertaria ≈99% das vezes sem detectar nenhuma fraude.

### Estrutura do projeto

```
Transaction-Fraud-ML/
├── MlFraude.py                  # script principal com todo o pipeline
├── requirements.txt
├── LICENSE
├── data/
│   ├── treino.csv                # dados de treino
│   ├── teste.csv                 # holdout sem rótulo (estilo Kaggle)
│   └── submissao.csv             # previsões geradas pelo modelo final
├── reports/
│   └── metrics.json              # métricas reais geradas pela última execução
└── notebooks/
    └── Sistema_de_Fraude.ipynb   # versão em notebook do mesmo pipeline
```

### Tecnologias

- [Python 3.13](https://www.python.org/)
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — manipulação de dados
- [scikit-learn](https://scikit-learn.org/) — modelos, pré-processamento, `RandomizedSearchCV` e métricas
- [Jupyter](https://jupyter.org/) — notebook exploratório

### Como executar

```bash
git clone git@github.com:Fefevs09/Transaction-Fraud-ML.git
cd Transaction-Fraud-ML

python3.13 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python MlFraude.py
```

> Os dados já estão disponíveis localmente em `data/`, mas por padrão o script carrega `treino.csv`/`teste.csv` a partir de uma URL raw do GitHub deste mesmo repositório (garante que o dado usado é sempre o que está versionado no branch `main`).

Alternativamente, o mesmo pipeline pode ser explorado interativamente em `notebooks/Sistema_de_Fraude.ipynb`.

### Metodologia

1. **Limpeza**: remoção de colunas de PII e alta cardinalidade que não agregam sinal ao modelo (`ssn`, `cc_num`, `first`, `last`, `street`, `zip`, `unix_time`, `lat`, `long`, `acct_num`, `trans_num`, `merchant`, `merch_lat`, `merch_long`).
2. **Engenharia de features temporais**: extração de `hour`, `minute`, `seconds`, `year`, `month`, `day` a partir de `trans_date`/`trans_time`.
3. **Idade do titular**: cálculo de `age` a partir de `dob` e do ano da transação.
4. **Encoding categórico**: `LabelEncoder` aplicado a `gender`, `job`, `profile`, `category`, `city`, `state`.
5. **Split treino/teste**: 80/20, estratificado pela variável alvo (`stratify=y`) para preservar a proporção de fraude em ambos os conjuntos.
6. **Imputação**: `SimpleImputer` ajustado no treino e aplicado ao teste.
7. **Comparação de modelos baseline**: KNN, Regressão Logística e Random Forest, todos avaliados com as mesmas métricas.
8. **Ajuste de hiperparâmetros**: `RandomizedSearchCV` (`cv=5`, `n_iter=20`) sobre o Random Forest, buscando `n_estimators`, `max_depth`, `min_samples_split` e `min_samples_leaf`.
9. **Inferência final**: o melhor modelo é aplicado ao holdout `teste.csv`, gerando `data/submissao.csv`.

### Resultados

Métricas reais, obtidas no conjunto de teste interno (20% de `treino.csv`, split estratificado), disponíveis em [`reports/metrics.json`](reports/metrics.json):

| Modelo | Accuracy | Precision (fraude) | Recall (fraude) | F1 (fraude) | ROC-AUC |
|---|---|---|---|---|---|
| KNN | 99,13% | 67,28% | 35,62% | 46,58% | 0,841 |
| Regressão Logística | 98,80% | 0,00% | 0,00% | 0,00% | 0,839 |
| Random Forest (padrão) | 99,67% | 97,73% | 70,26% | 81,75% | 0,992 |
| **Random Forest (ajustado)** | 99,64% | 97,20% | 67,97% | 80,00% | **0,997** |

Melhores hiperparâmetros encontrados pelo `RandomizedSearchCV` (`cv=5`, `n_iter=20`): `n_estimators=510`, `max_depth=None`, `min_samples_split=14`, `min_samples_leaf=1`.

Alguns pontos honestos sobre esses números:
- A **Regressão Logística** atinge quase 99% de acurácia sem detectar uma única fraude (precision e recall = 0) — o exemplo mais claro de por que acurácia isolada é enganosa neste problema.
- O **Random Forest com parâmetros padrão** já é competitivo; o `RandomizedSearchCV` (que otimiza acurácia por padrão, já que nenhum `scoring` foi especificado) melhora o **ROC-AUC** (0,992 → 0,997), mas não superou o F1/recall do modelo padrão nesta busca — um lembrete de que tuning automatizado nem sempre melhora todas as métricas ao mesmo tempo, e que o critério de otimização (`scoring`) deveria ser escolhido de acordo com o que importa para o negócio (ex.: `scoring="f1"` ou `scoring="recall"` para priorizar a detecção de fraude).

Como o dataset é ≈99% legítimo / ≈1% fraude, **accuracy não é uma métrica confiável** — um classificador que sempre prevê "não fraude" já atingiria ≈99% de acurácia sem identificar um único caso de fraude. Por isso as métricas centrais aqui são **precision, recall e F1 focadas na classe fraude**, além de **ROC-AUC**: elas mostram o quão bem o modelo realmente distingue fraude de transações legítimas, e o trade-off entre detectar fraude (recall) e evitar falsos alarmes (precision).

### Limitações e próximos passos

- O pipeline original avaliava os modelos apenas por `.score()` (acurácia), o que é enganoso neste dataset desbalanceado — corrigido nesta atualização com precision/recall/F1/ROC-AUC focados na classe fraude.
- O desbalanceamento de classes não é tratado ativamente (sem SMOTE, undersampling ou `class_weight='balanced'`) — candidato natural para trabalho futuro.
- Há uma inconsistência conhecida no bloco final de inferência: o `LabelEncoder` das colunas categóricas é reajustado (`fit_transform`) em `teste.csv` em vez de reutilizar o encoder ajustado em `treino.csv`. Isso afeta apenas as previsões finais em `data/submissao.csv`, não as métricas de avaliação reportadas acima (que vêm do split interno de `treino.csv`). Documentado aqui, não corrigido nesta passada.
- Código morto residual no script (import não usado de `sklearn.neighbors`, `StandardScaler`/bloco `OneHotEncoder` comentado, função `set_street_num` não utilizada).
- Sem testes automatizados ou CI configurados.
- `notebooks/Sistema_de_Fraude.ipynb` espelha o script, mas pode ficar levemente fora de sincronia após mudanças futuras em `MlFraude.py`.

### Licença

Distribuído sob a licença MIT — veja [LICENSE](./LICENSE).

### Autor

Desenvolvido por [Felipe Fernandes (@Fefevs09)](https://github.com/Fefevs09).

---

## 🇺🇸 English

### Table of Contents
- [About](#about)
- [Dataset](#dataset-1)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)
- [How to run](#how-to-run)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations & future work](#limitations--future-work)
- [License](#license)
- [Author](#author)

### About

This is a study/portfolio project that implements a full credit card transaction fraud detection pipeline: data cleaning and feature engineering, comparison of classification models (KNN, Logistic Regression and Random Forest), and hyperparameter tuning via `RandomizedSearchCV`.

Fraud detection is a classic **extremely imbalanced classification** problem — the vast majority of transactions are legitimate, and that's exactly what makes choosing the right evaluation metrics so important (see [Results](#results)).

### Dataset

The data follows the schema of the [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection), a set of **synthetic** transactions generated by the [Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation) simulator, widely used in fraud-detection studies.

| Column | Description |
|---|---|
| `index` | Unique identifier for each row |
| `trans_date_trans_time` | Transaction date and time |
| `cc_num` | Customer's credit card number |
| `merchant` | Merchant name |
| `category` | Merchant category |
| `amt` | Transaction amount |
| `first` / `last` | Cardholder first / last name |
| `gender` | Cardholder gender |
| `street` / `city` / `state` / `zip` | Cardholder home address |
| `lat` / `long` | Cardholder latitude / longitude |
| `city_pop` | Population of the cardholder's city |
| `job` | Cardholder occupation |
| `dob` | Cardholder date of birth |
| `trans_num` | Transaction number |
| `unix_time` | Transaction UNIX timestamp |
| `merch_lat` / `merch_long` | Merchant latitude / longitude |
| `is_fraud` | Fraud indicator — **target class** |

**Class imbalance** (measured directly from `data/treino.csv`):

| Set | Rows | Legitimate | Fraud |
|---|---|---|---|
| Train (`treino.csv`) | 144,353 | 142,666 (≈99.06%) | 1,528 (≈1.06%) |
| Test (`teste.csv`) | 36,048 | unlabeled — Kaggle-style holdout, predicted into `data/submissao.csv` | |

This imbalance is exactly why plain accuracy is misleading here: a trivial "always predict not-fraud" classifier would already score ≈99% accuracy while catching zero fraud.

### Project structure

```
Transaction-Fraud-ML/
├── MlFraude.py                  # main script with the full pipeline
├── requirements.txt
├── LICENSE
├── data/
│   ├── treino.csv                # training data
│   ├── teste.csv                 # unlabeled holdout (Kaggle-style)
│   └── submissao.csv             # predictions from the final model
├── reports/
│   └── metrics.json              # real metrics from the latest run
└── notebooks/
    └── Sistema_de_Fraude.ipynb   # notebook version of the same pipeline
```

### Tech stack

- [Python 3.13](https://www.python.org/)
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data manipulation
- [scikit-learn](https://scikit-learn.org/) — models, preprocessing, `RandomizedSearchCV` and metrics
- [Jupyter](https://jupyter.org/) — exploratory notebook

### How to run

```bash
git clone git@github.com:Fefevs09/Transaction-Fraud-ML.git
cd Transaction-Fraud-ML

python3.13 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python MlFraude.py
```

> Data is already available locally under `data/`, but by default the script loads `treino.csv`/`teste.csv` from a raw GitHub URL pointing at this same repository (guarantees the data used always matches what's versioned on `main`).

Alternatively, the same pipeline can be explored interactively in `notebooks/Sistema_de_Fraude.ipynb`.

### Methodology

1. **Cleaning**: drop PII and high-cardinality columns that add no modeling signal (`ssn`, `cc_num`, `first`, `last`, `street`, `zip`, `unix_time`, `lat`, `long`, `acct_num`, `trans_num`, `merchant`, `merch_lat`, `merch_long`).
2. **Time-based feature engineering**: extract `hour`, `minute`, `seconds`, `year`, `month`, `day` from `trans_date`/`trans_time`.
3. **Cardholder age**: compute `age` from `dob` and the transaction year.
4. **Categorical encoding**: `LabelEncoder` applied to `gender`, `job`, `profile`, `category`, `city`, `state`.
5. **Train/test split**: 80/20, stratified on the target (`stratify=y`) to preserve the fraud ratio in both sets.
6. **Imputation**: `SimpleImputer` fit on train, applied to test.
7. **Baseline model comparison**: KNN, Logistic Regression and Random Forest, all evaluated with the same metrics.
8. **Hyperparameter tuning**: `RandomizedSearchCV` (`cv=5`, `n_iter=20`) over Random Forest, searching `n_estimators`, `max_depth`, `min_samples_split` and `min_samples_leaf`.
9. **Final inference**: the best model is applied to the `teste.csv` holdout, producing `data/submissao.csv`.

### Results

Real metrics, measured on the internal test set (20% of `treino.csv`, stratified split), available in [`reports/metrics.json`](reports/metrics.json):

| Model | Accuracy | Precision (fraud) | Recall (fraud) | F1 (fraud) | ROC-AUC |
|---|---|---|---|---|---|
| KNN | 99.13% | 67.28% | 35.62% | 46.58% | 0.841 |
| Logistic Regression | 98.80% | 0.00% | 0.00% | 0.00% | 0.839 |
| Random Forest (default) | 99.67% | 97.73% | 70.26% | 81.75% | 0.992 |
| **Random Forest (tuned)** | 99.64% | 97.20% | 67.97% | 80.00% | **0.997** |

Best hyperparameters found by `RandomizedSearchCV` (`cv=5`, `n_iter=20`): `n_estimators=510`, `max_depth=None`, `min_samples_split=14`, `min_samples_leaf=1`.

A few honest observations on these numbers:
- **Logistic Regression** reaches almost 99% accuracy while catching zero fraud cases (precision and recall = 0) — the clearest example of why accuracy alone is misleading on this problem.
- The **default-parameter Random Forest** is already competitive; `RandomizedSearchCV` (which optimizes accuracy by default, since no `scoring` was specified) improves **ROC-AUC** (0.992 → 0.997) but doesn't beat the default model's F1/recall in this particular search — a reminder that automated tuning doesn't always improve every metric at once, and the optimization criterion (`scoring`) should be chosen based on what actually matters for the business (e.g. `scoring="f1"` or `scoring="recall"` to prioritize catching fraud).

Since the dataset is ≈99% legitimate / ≈1% fraud, **accuracy is not a trustworthy metric** — a classifier that always predicts "not fraud" would already reach ≈99% accuracy without catching a single fraud case. That's why the metrics that matter here are **precision, recall and F1 focused on the fraud class**, plus **ROC-AUC**: they show how well the model actually separates fraud from legitimate transactions, and the trade-off between catching fraud (recall) and avoiding false alarms (precision).

### Limitations & future work

- The original pipeline only evaluated models with `.score()` (accuracy), which is misleading on this imbalanced dataset — fixed in this update with precision/recall/F1/ROC-AUC focused on the fraud class.
- Class imbalance is not actively addressed (no SMOTE, undersampling, or `class_weight='balanced'`) — a natural candidate for future work.
- There's a known inconsistency in the final inference block: the categorical `LabelEncoder` is refit (`fit_transform`) on `teste.csv` instead of reusing the encoder fit on `treino.csv`. This only affects the final predictions in `data/submissao.csv`, not the evaluation metrics reported above (which come from the internal `treino.csv` split). Documented here, not fixed in this pass.
- Some residual dead code in the script (unused `sklearn.neighbors` import, commented-out `StandardScaler`/`OneHotEncoder` block, unused `set_street_num` function).
- No automated tests or CI configured.
- `notebooks/Sistema_de_Fraude.ipynb` mirrors the script but may drift slightly out of sync after future changes to `MlFraude.py`.

### License

Distributed under the MIT license — see [LICENSE](./LICENSE).

### Author

Built by [Felipe Fernandes (@Fefevs09)](https://github.com/Fefevs09).
