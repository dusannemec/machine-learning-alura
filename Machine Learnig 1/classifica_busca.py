import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from joblib import dump, load
from collections import Counter
from fit_and_predict import fit_and_predict

data_frame = pd.read_csv('buscas2.csv')

X_df = data_frame[['home', 'busca', 'logado']]
Y_df = data_frame['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_treino = 0.8
porcentagem_teste = 0.1

tamanho_treino = int(porcentagem_treino * len(Y))
tamanho_teste = int(porcentagem_teste * len(Y))
tamanho_validacao = len(Y) -  tamanho_treino - tamanho_teste

treino_dados = X[:tamanho_treino]
treino_maracacoes = Y[:tamanho_treino]

teste_dados = X[tamanho_treino:tamanho_treino+tamanho_teste]
teste_marcacoes = Y[tamanho_treino:tamanho_treino+tamanho_teste]
fim_teste = tamanho_treino + tamanho_teste

validacao_dados = X[fim_teste:]
validacao_marcacoes = Y[fim_teste:]

#Treina e salva:
IA_MNB = MultinomialNB()
resultado_MNB = fit_and_predict(IA_MNB, 'MultinomialNB', treino_dados, treino_maracacoes, teste_dados, teste_marcacoes)
IA_Ada = AdaBoostClassifier()
resultado_Ada = fit_and_predict(IA_Ada, 'AdaBoostClassifier', treino_dados, treino_maracacoes, teste_dados, teste_marcacoes)

vencedor = IA_MNB if resultado_MNB > resultado_Ada else IA_Ada

print(f'Validando vencedor: {vencedor}')
previsao = vencedor.predict(validacao_dados)
print(f'Previ: {len(previsao)}')

diferencas = previsao - validacao_marcacoes 
acertos = [d for d in diferencas if d == 0]
print(f'Acertei: {len(acertos)}')

porcentagem_acerto = len(acertos) * 100 / len(validacao_marcacoes)
print(f'Pocentagem de acerto: {porcentagem_acerto:.2f}%')
print('------------------------------------------------')

# algoritmo base
# acerto_de_um = len(Y[Y=='sim'])
# acerto_de_zero = len(Y[Y=='nao'])
taxa_de_acerto_base = 100 * max(Counter(validacao_marcacoes).values()) / len(validacao_marcacoes)
print(f'Taxa de acerto base: {taxa_de_acerto_base:.2f}%')
