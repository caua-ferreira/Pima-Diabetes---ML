# Pima Diabetes Pipeline (pandas + scikit-learn)

Repositório com um pipeline simples para o dataset Pima Indians Diabetes (UCI). O objetivo é fornecer um projeto "plug-and-play" para treinar e avaliar modelos localmente sem necessidade de Spark/Java.

<img width="2050" height="2078" alt="image" src="https://github.com/user-attachments/assets/9222025b-e269-4d9f-916f-dcf424ba0d16" />



# Pima Diabetes Pipeline (pandas + scikit-learn)

Repositório com um pipeline simples para o dataset Pima Indians Diabetes (UCI). O objetivo é fornecer um projeto "plug-and-play" para treinar e avaliar modelos localmente e usar o modelo treinado para prever em amostras reais (ex.: `voluntarios.json`).

**Conteúdo principal**
- `pima_diabetes_pipeline.py`: script para treinar/eavaliar modelos e salvar o melhor pipeline em `best_model.joblib`.
- `best_model.joblib`: arquivo com o pipeline/modelo treinado (se presente).
- `voluntarios.json`: amostra de entrada com voluntários de teste.
- `example/predict_voluntarios.py`: exemplo para carregar `voluntarios.json` e gerar previsões usando o modelo salvo.
- `requirements.txt`: dependências Python.

**Preparação (local)**
1. Crie e ative um ambiente virtual (recomendado):
```
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```
2. Instale dependências:
```
pip install -r requirements.txt
```

**Treinar o modelo**
```
python pima_diabetes_pipeline.py --model-out best_model.joblib
```

Isso fará download do dataset (UCI/GitHub), fará o pré-processamento (substitui zeros inválidos por `NaN` e imputa pela mediana), treinará modelos (Logistic Regression e RandomForest via GridSearch) e salvará o melhor pipeline em `best_model.joblib`.

**Usar o modelo para prever (exemplo)**
```
python example/predict_voluntarios.py --model best_model.joblib --input voluntarios.json --out predictions.csv
```

O script `example/predict_voluntarios.py` aplica o mesmo pré-processamento do pipeline (imputação de zeros) e gera `predictions.csv` com colunas `id`, `prediction` e `probability`.

**Sobre arquivos grandes**
- Se `best_model.joblib` for grande, recomenda-se usar Git LFS para versionar o arquivo (`git lfs install; git lfs track "*.joblib"`). Este repositório contém `.gitattributes` sugerindo LFS.

**Boas práticas antes de subir**
- Não commite a pasta `desenvolvimento/` (ambient virtual). Já incluí `.gitignore` para isso.

**Licença**
Projeto sob licença MIT — ver `LICENSE`.

Contribuições são bem-vindas.
