# NeuroBrasil — Rede Neural do Zero

Este projeto foi desenvolvido para uma atividade acadêmica de **Redes Neurais Artificiais**. A ideia é mostrar, de forma prática e visual, como uma rede neural aprende a fazer uma classificação sem usar TensorFlow, Keras, PyTorch ou scikit-learn.

O projeto possui duas partes:

1. **Parte de Machine Learning:** carregamento dos dados, preparação, rede neural, treinamento e métricas.
2. **Parte visual:** uma aplicação web simples em Flask que apresenta a arquitetura e os resultados.

---

## 1. Qual problema o projeto resolve?

O modelo analisa dados estaduais de COVID-19 no Brasil e responde à seguinte pergunta:

> Neste registro, o número de novos óbitos está entre os 25% maiores valores observados no conjunto de dados?

Essa é uma tarefa de **classificação binária**, porque existem somente duas respostas:

- `0` — não é um dia de alta incidência de óbitos;
- `1` — é um dia de alta incidência de óbitos.

É importante entender que o projeto não tenta prever exatamente quantas pessoas morrerão. Ele classifica o registro como **baixo/normal** ou **alto** em relação ao limite calculado nos dados.

### Por que esse problema foi escolhido?

- O dataset é relacionado exclusivamente ao Brasil.
- Os dados são reais e públicos.
- O problema é adequado para uma primeira rede neural.
- É possível transformar o resultado em uma classificação objetiva.
- O dashboard facilita a visualização do aprendizado da rede.

---

## 2. Qual dataset foi utilizado?

O arquivo utilizado é o `cases-brazil-states.csv`, do projeto **COVID-19 Brasil**, mantido por William Cota, com registros por estado brasileiro e referências a fontes oficiais.

Fonte pública:

<https://github.com/wcota/covid19br>

O código não exige que o CSV seja baixado manualmente. Na primeira execução, o arquivo é baixado automaticamente para:

```text
data/cases-brazil-states.csv
```

Nas execuções seguintes, o arquivo local é reaproveitado.

### Principais colunas usadas

| Coluna | Significado |
|---|---|
| `newCases` | novos casos registrados no dia anterior |
| `newDeaths` | novos óbitos registrados no dia |
| `totalCases` | total acumulado de casos |
| `deaths` | total acumulado de óbitos |
| `recovered` | total de recuperados |
| `date` | data do registro |
| `state` | estado brasileiro |

O arquivo atual não fornece diretamente `activeCases`. Por isso, o projeto calcula:

```text
activeCases = totalCases - deaths - recovered
```

Quando o resultado seria negativo por alguma inconsistência do dado, ele é ajustado para zero.

---

## 3. Como os dados viram entradas da rede?

Para cada estado, o código ordena os registros pela data. Depois, ele utiliza informações do dia anterior como entradas:

1. `previous_cases`: casos novos do dia anterior;
2. `previous_deaths`: óbitos novos do dia anterior;
3. `previous_active`: casos ativos do dia anterior;
4. `day_number`: quantidade de dias desde o primeiro registro.

Usar os valores anteriores é importante porque evita entregar diretamente para a rede a informação que queremos classificar. Se o modelo recebesse `newDeaths` do próprio dia como entrada, haveria **vazamento de informação** (*data leakage*).

### Como o rótulo é criado?

O código calcula o percentil 75 de `newDeaths`:

```python
threshold = frame["newDeaths"].quantile(0.75)
```

Depois, cada registro recebe um rótulo:

```python
frame["target"] = (frame["newDeaths"] >= threshold).astype(int)
```

Assim, registros com óbitos maiores ou iguais ao limite recebem `1`; os demais recebem `0`.

---

## 4. Pré-processamento

Antes do treinamento, o pipeline realiza estas etapas:

1. baixa ou lê o CSV;
2. converte a data para o formato correto;
3. converte colunas numéricas;
4. calcula `activeCases`;
5. ordena os dados por estado e data;
6. cria as colunas referentes ao dia anterior;
7. remove linhas sem dados suficientes;
8. cria o alvo `target`;
9. separa treino e teste;
10. normaliza as entradas.

### Divisão entre treino e teste

O projeto usa:

- **80%** dos registros para treinamento;
- **20%** para avaliação.

A divisão é feita antes da normalização. A média e o desvio padrão são calculados somente no conjunto de treino:

```python
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
```

Isso é importante porque o conjunto de teste deve permanecer desconhecido para o modelo durante o aprendizado.

---

## 5. Como funciona a rede neural?

A rede foi implementada manualmente no arquivo [neural_network.py](./neural_network.py), usando somente NumPy para os cálculos matriciais.

Sua arquitetura é:

```text
4 entradas → 12 neurônios → 8 neurônios → 1 saída
```

No código, ela é criada assim:

```python
network = NeuralNetwork(
    [4, 12, 8, 1],
    learning_rate=0.008,
    seed=7
)
```

### O que significa cada camada?

- **4 entradas:** as quatro características preparadas no pipeline;
- **12 neurônios:** primeira camada oculta, que aprende combinações das entradas;
- **8 neurônios:** segunda camada oculta, que refina essas combinações;
- **1 saída:** probabilidade de o registro pertencer à classe `1`.

### Pesos e bias

Cada conexão possui um peso. Os pesos indicam a importância de cada entrada. Cada neurônio também possui um `bias`, que ajuda a ajustar o limite de decisão.

Os pesos começam com valores aleatórios pequenos usando inicialização He:

```python
sqrt(2 / quantidade_de_entradas)
```

O `seed=7` faz com que o início aleatório seja reproduzível.

---

## 6. Feedforward: a rede fazendo uma previsão

No feedforward, os dados atravessam a rede da esquerda para a direita.

Para cada camada, a rede calcula:

```text
z = entrada × pesos + bias
```

Nas camadas ocultas é aplicada a função ReLU:

```python
relu(z) = max(0, z)
```

A ReLU mantém valores positivos e transforma valores negativos em zero. Ela ajuda a rede a aprender relações não lineares.

Na última camada é usada a Sigmoid:

```python
sigmoid(z) = 1 / (1 + e^(-z))
```

A Sigmoid transforma o resultado em um número entre `0` e `1`, que pode ser interpretado como probabilidade.

No final:

```python
prediction = network.predict(x_test)
```

Se a probabilidade for maior ou igual a `0.5`, a previsão vira `1`; caso contrário, vira `0`.

---

## 7. Função de perda

O projeto usa **Binary Cross-Entropy**, adequada para classificação binária.

Ela compara o valor verdadeiro com a probabilidade prevista:

```python
loss = -mean(
    y_true * log(y_pred)
    + (1 - y_true) * log(1 - y_pred)
)
```

A interpretação é simples:

- previsão correta e confiante: perda pequena;
- previsão errada e confiante: perda grande;
- durante o treinamento, esperamos que a perda diminua.

O gráfico no dashboard mostra justamente essa evolução.

---

## 8. Backpropagation

O backpropagation é o processo que faz a rede aprender.

Em cada lote de dados, o programa:

1. faz uma previsão;
2. compara a previsão com a resposta correta;
3. calcula o erro da saída;
4. propaga o erro para as camadas anteriores;
5. calcula os gradientes dos pesos e biases;
6. atualiza os parâmetros.

A atualização segue a ideia:

```text
novo_peso = peso_atual - taxa_de_aprendizado × gradiente
```

No projeto, isso aparece em:

```python
self.weights[index] -= self.learning_rate * gradients_w[index]
self.biases[index] -= self.learning_rate * gradients_b[index]
```

### Taxa de aprendizado

A taxa escolhida foi `0.008`. Ela controla o tamanho de cada ajuste:

- muito grande: a rede pode passar do ponto ideal;
- muito pequena: o treinamento fica muito lento;
- `0.008`: valor definido para manter uma evolução estável neste dataset.

### Épocas e mini-batches

O treinamento usa:

- `260` épocas;
- lotes de `64` registros.

Uma época significa que a rede passou por todos os dados de treino uma vez. O uso de mini-batches evita atualizar os pesos somente depois de processar o dataset inteiro.

---

## 9. Métricas

Depois do treinamento, a rede é avaliada no conjunto de teste.

### Acurácia

Percentual total de previsões corretas:

```text
(verdadeiros positivos + verdadeiros negativos) / total
```

### Precisão

Entre os registros classificados como alta incidência, quantos realmente eram alta incidência:

```text
verdadeiros positivos / (verdadeiros positivos + falsos positivos)
```

### Recall

Entre todos os registros que realmente eram de alta incidência, quantos a rede encontrou:

```text
verdadeiros positivos / (verdadeiros positivos + falsos negativos)
```

### F1-Score

É a média harmônica entre precisão e recall. É útil quando queremos equilibrar os dois:

```text
F1 = 2 × precisão × recall / (precisão + recall)
```

### Matriz de confusão

O código também calcula:

- `TP`: previu alta incidência e estava correto;
- `TN`: previu baixa incidência e estava correto;
- `FP`: previu alta incidência, mas estava errado;
- `FN`: previu baixa incidência, mas deveria ser alta.

---

## 10. Como os arquivos se conectam?

### [data_pipeline.py](./data_pipeline.py)

Responsável pelos dados:

- download do CSV;
- limpeza;
- criação das entradas;
- normalização;
- separação de treino e teste.

### [neural_network.py](./neural_network.py)

Contém a implementação da rede:

- pesos;
- biases;
- ReLU;
- Sigmoid;
- feedforward;
- loss;
- backpropagation;
- treinamento;
- previsão.

### [app.py](./app.py)

É o servidor Flask. Ele:

- abre a página inicial;
- chama o pipeline;
- cria e treina a rede;
- calcula as métricas;
- envia os resultados para o frontend em JSON.

O endpoint principal de treinamento é:

```text
GET /api/train
```

### [templates/index.html](./templates/index.html)

Define a estrutura da página: título, cards, arquitetura e área do gráfico.

### [static/style.css](./static/style.css)

Define o visual: cores, espaçamentos, tipografia, responsividade e cards.

### [static/app.js](./static/app.js)

Conecta o botão à API, recebe os resultados e atualiza:

- acurácia;
- precisão;
- F1-Score;
- quantidade de amostras;
- data mais recente;
- gráfico da perda.

---

## 11. Como executar no Windows

Abra o PowerShell na pasta do projeto:

```powershell
cd "C:\Users\dudas\OneDrive\Desktop\rede neural - atv 2"
```

Opcionalmente, crie um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

Inicie o servidor:

```powershell
python app.py
```

Abra no navegador:

<http://127.0.0.1:5000>

Na página, clique em **“Treinar a rede”**. O sistema baixará os dados na primeira vez, executará o treinamento e mostrará os resultados.

---

## 12. Resultado observado

Durante a validação do projeto, o pipeline carregou:

- 12.585 amostras de treinamento;
- 3.147 amostras de teste.

O endpoint de treinamento foi executado com sucesso e retornou F1-Score de aproximadamente `0.91` naquela execução.

Esse valor pode variar levemente entre execuções porque os mini-batches são embaralhados durante o treinamento. Ele não deve ser interpretado como um sistema médico ou como uma previsão oficial; trata-se de uma demonstração acadêmica de uma rede neural construída do zero.
