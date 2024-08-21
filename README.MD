# ezneural

**ezneural** é uma biblioteca Python projetada para tornar a criação e treinamento de redes neurais mais simples e acessível. Ideal para iniciantes que desejam experimentar redes neurais sem a complexidade dos frameworks mais avançados.

## Funcionalidades

- Criação de camadas densas (Fully Connected)
- Suporte a funções de ativação ReLU, Sigmoid e Tanh
- Função de perda Mean Squared Error
- Otimizador SGD (Stochastic Gradient Descent)
- Interface intuitiva para construção e treinamento de redes neurais

### Camadas

- **`Dense`**: Camada densa totalmente conectada com funções de ativação ReLU, Sigmoid ou Tanh.

### Modelos

- **`Model`**: Interface para construir, compilar, treinar e prever com redes neurais.

### Otimizadores

- **`Optimizer`**: Classe base para otimizadores.
- **`SGD`**: Implementação do otimizador Stochastic Gradient Descent.

### Funções de Perda

- **`Loss`**: Classe base para funções de perda.
- **`MeanSquaredError`**: Implementação da função de perda Mean Squared Error.

## Instalação

Clone o repositório e instale as dependências necessárias:

```bash
git clone https://github.com/wanbnn/ezneural.git
cd ezneural
pip install numpy
```

Uso
Aqui está um exemplo básico de como usar a biblioteca:

```python
from ezneural.model import Model
from ezneural.layers import Dense
from ezneural.optimizers import SGD
from ezneural.utils import MeanSquaredError
import numpy as np

# Criar o modelo
model = Model()
model.add(Dense(2, 10, activation='relu'))
model.add(Dense(10, 1, activation='sigmoid'))

# Compilar o modelo
optimizer = SGD(learning_rate=0.01)
loss_function = MeanSquaredError()
model.compile(loss_function, optimizer)

# Dados fictícios
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Treinar o modelo
model.train(X_train, y_train, epochs=1000, batch_size=4)

# Fazer previsões
predictions = model.predict(X_train)
print(predictions)
```


Contribuições são bem-vindas! Sinta-se à vontade para enviar pull requests ou abrir issues para discutir melhorias.

Licença
Este projeto é licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
