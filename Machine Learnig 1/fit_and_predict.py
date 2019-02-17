#import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

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
	return porcentagem_acerto
