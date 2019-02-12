from dados import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

# minha abordagem inicial foi
#     1. separar 90% para treino e 10% para teste: 88.89%

X, Y = carregar_acessos()
modelo = MultinomialNB()

treino_dados = X[:90]
treino_marcacoes = Y[:90]

teste_dados = X[-9:]
teste_marcacoes = Y[-9:]

modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
print(f'Previ: {resultado}')

diferencas = resultado - teste_marcacoes
acertos = [d for d in diferencas if d == 0]
porcentagem_acerto = len(acertos) * 100 / len(teste_marcacoes)
print(f'Taxa de acerto: {porcentagem_acerto:.2f}%')
