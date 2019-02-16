import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from joblib import dump, load
from collections import Counter

def fit_and_predict(IA, nome, treino_dados, treino_maracacoes, teste_dados, teste_marcacoes):
	print(nome)
	IA.fit(treino_dados, treino_maracacoes)

	#dump(IA, 'IA.joblib')
	#IA = load('IA.joblib')

	previsao = IA.predict(teste_dados)
	print(f'Previ: {len(previsao)}')

	diferencas = previsao - teste_marcacoes
	#acertos = sum(previsao == teste_marcacoes) 
	acertos = [d for d in diferencas if d == 0]
	print(f'Acertei: {len(acertos)}')

	porcentagem_acerto = len(acertos) * 100 / len(teste_marcacoes)
	print(f'Pocentagem de acerto: {porcentagem_acerto:.2f}%')
	print('------------------------------------------------')


data_frame = pd.read_csv('busca.csv')

X_df = data_frame[['home', 'busca', 'logado']]
Y_df = data_frame['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_treino = 0.9

tamanho_treino = int(porcentagem_treino * len(Y))
tamanho_teste = len(Y) - tamanho_treino

treino_dados = X[:tamanho_treino]
treino_maracacoes = Y[:tamanho_treino]

teste_dados = X[-tamanho_teste:]
teste_marcacoes = Y[-tamanho_teste:]

#Treina e salva:
IA_MNB = MultinomialNB()
fit_and_predict(IA_MNB, 'MultinomialNB', treino_dados, treino_maracacoes, teste_dados, teste_marcacoes)
IA_Ada = AdaBoostClassifier()
fit_and_predict(IA_Ada, 'AdaBoostClassifier', treino_dados, treino_maracacoes, teste_dados, teste_marcacoes)

# algoritmo base
# acerto_de_um = len(Y[Y=='sim'])
# acerto_de_zero = len(Y[Y=='nao'])
taxa_de_acerto_base = 100 * max(Counter(teste_marcacoes).values()) / len(teste_marcacoes)
print(f'Taxa de acerto base: {taxa_de_acerto_base:.2f}%')
