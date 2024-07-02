## XAI to Stress Identification
Arquivo analysis.ipynb
1- Decision Tree
2- Random Forest
3- Feature Importance
4- Shap - Gráfico de importância de variáveis
        - Gráfico SHAP value para cada classe do dataset (Not stressed, middle value, stressed)
5- Lime - Gráfico de importância de variáveis para cada classe para uma instância em específico
6- XGBoost

## WESAD
Análise do dataset WESAD e table.md com tabelas de algoritmos usados
### Preparação do Dataset
1- Importar dados (bvp, eda, temp e label do arquivo pkl e hr do HR.csv)
2- Interpolação usando linspace e interp1d para lidar com as diferenças de sample rate
3- Normalização dos valores
4- Criação de colunas min, max e mean (a cada 3 linhas achar o valor min, max e mean)

### Aplicação de algoritmos de classificação
1- Algoritmos aplicados: KNN, Logistic Regression, Random Forest, Neural Network, SVM
    Mesmos resultados obtidos da Milena
2- Gráfico de Feature Importance

### Criação de Dataset individual
1- Importar dados (bvp, eda, temp e label do arquivo pkl e hr do HR.csv)
2- Interpolação usando linspace e interp1d para lidar com as diferenças de sample rate
3- Normalização dos valores
4- Criação de colunas min, max e mean (a cada 3 linhas achar o valor min, max e mean)

### Aplicação de séries temporais
1- VAR: obteve bons resultados
2- Time Series Forecast 
3- Rocket 
4- CNN Regressor

### Uso de algoritmos de classificação já treinados em datasets individuais
Random Forest obteve o pior resultado
