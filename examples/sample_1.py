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