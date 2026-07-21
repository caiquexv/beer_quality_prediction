# Previsão da Qualidade de Cervejas com Métodos de Otimização

Projeto acadêmico que aplica **regressão linear** e **algoritmos de otimização** para prever a qualidade de cervejas a partir de suas características físico-químicas.

## 📌 Sobre o projeto

O objetivo principal é desenvolver modelos preditivos capazes de avaliar a qualidade de cervejas com base em variáveis físico-químicas (como temperatura, acidez e outras medidas do processo de fabricação). Utilizando técnicas de regressão linear combinadas com algoritmos de otimização, o projeto busca identificar quais combinações de variáveis impactam de forma mais significativa a qualidade final da bebida.

## 🧮 Metodologia

O projeto implementa e compara diferentes abordagens de otimização para o ajuste dos coeficientes da regressão:

- **Função de perda (MSE)**: cálculo do erro quadrático médio entre as predições e os valores reais.
- **Gradiente da MSE**: cálculo analítico do gradiente da função de perda em relação aos coeficientes, usado para acelerar a convergência dos métodos de otimização.
- **Otimização via SciPy (BFGS)**: uso do solver `scipy.optimize.minimize` com o método BFGS, utilizando o gradiente analítico como jacobiano.
- **Gradiente descendente implementado do zero**: uma versão própria do algoritmo de gradiente descendente, sem depender de bibliotecas de otimização prontas — útil para fins didáticos e de comparação com o resultado do SciPy.
- **Seleção de variáveis**: testes com diferentes combinações de colunas de entrada, buscando o subconjunto de variáveis que gera o melhor ajuste do modelo.

## 🛠️ Tecnologias utilizadas

- **Python 3**
- **NumPy** — operações matriciais e cálculos numéricos
- **Pandas** — carregamento e manipulação dos dados
- **SciPy** (`scipy.optimize.minimize`) — otimização via método BFGS

## 📂 Estrutura do repositório

```
.
├── beer_quality_prediction.py   # Script principal do projeto
├── cerveja_treino.csv         # Dados de treino
├── cerveja_teste.csv          # Dados de teste
├── .gitignore
└── README.md
```

## 📊 Sobre os dados

Os arquivos `cerveja_treino.csv` e `cerveja_teste.csv` contêm variáveis físico-químicas (sem cabeçalho nomeado) usadas para treinar e avaliar o modelo.

> **Nota técnica**: os arquivos usam vírgula como separador decimal (padrão brasileiro), com os valores entre aspas (ex: `"21,8"`). Ao carregar com Pandas, use:
> ```python
> df = pd.read_csv("Cerveja_-_treino.csv", header=None, decimal=",")
> ```

## ▶️ Como executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/nome-do-repo.git
   cd nome-do-repo
   ```
2. Instale as dependências:
   ```bash
   pip install numpy pandas scipy
   ```
3. Execute o script:
   ```bash
   python beer_quality_prediction.py
   ```
   O script imprime o progresso no terminal e gera automaticamente os arquivos de previsão (`previsoes_R*_scipy.txt`, `previsoes_R*_gradiente.txt`) e um relatório final (`relatorio_final.txt`).

## 📄 Como interpretar o `relatorio_final.txt`

O relatório resume, para cada quantidade de variáveis testada, qual foi a melhor combinação encontrada e o quão bem o modelo se saiu. O próprio arquivo já traz um glossário explicando os termos, mas os pontos principais são:

- **R**: número de variáveis usadas no modelo (o script testa combinações de 1 a 4 variáveis dentre as 7 disponíveis, e guarda a melhor combinação de cada tamanho).
- **MSE (Erro Quadrático Médio)**: mede o quão distante as previsões do modelo estão dos valores reais. **Quanto menor, melhor.**
- **R²**: mede a proporção da variação dos dados que o modelo consegue explicar, numa escala de 0 a 1. **Quanto mais perto de 1, melhor** o ajuste aos dados de treino.
- **Scipy vs. Gradiente**: os coeficientes do modelo são calculados de duas formas — usando a biblioteca SciPy e usando um gradiente descendente implementado do zero. Quando os dois métodos chegam a resultados parecidos, é um bom sinal de que a implementação própria está correta.
- **Atenção ao overfitting**: usar mais variáveis quase sempre reduz o MSE no treino, mas isso pode significar que o modelo só "decorou" os dados, em vez de aprender um padrão que generaliza bem. Por isso, vale comparar se o ganho de um R para o próximo é significativo ou apenas marginal, em vez de simplesmente escolher o modelo com mais variáveis.

