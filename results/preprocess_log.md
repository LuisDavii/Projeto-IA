# Preprocessing log

## Zoo

- Arquivo de entrada: `data/raw/zoo.csv`
- Arquivo tratado: `data/processed/zoo.csv`
- Linhas antes/depois: 101/101
- Colunas antes/depois: 18/17
- Nulos na entrada: 0
- Nulos no CSV após a limpeza: 0
- Linhas removidas por alvo ausente: 0
- Linhas duplicadas removidas: 0
- Nulos em atributos preservados para imputação no Pipeline: 0
- Nulos imputados fora do Pipeline: 0 (imputação deve ser ajustada em cada fold).
- Colunas descartadas: animal_name.
- Outliers: nenhum valor foi removido/capado; os atributos são binários ou discretos e os valores observados de `legs` são válidos no domínio Zoo.
- Atributos categóricos restantes: nenhum; OneHotEncoder é aplicado dentro do Pipeline.
- StandardScaler e SimpleImputer são ajustados somente nos dados de treino de cada fold.
