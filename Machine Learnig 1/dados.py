import csv

def carregar_acessos():
    arquivo = open('acesso.csv', 'r')
    leitor = csv.reader(arquivo)

    next(leitor)

    X = []
    Y = []

    for home, como_funciona, contato, comprou in leitor:
        dados = [int(home), int(como_funciona), int(contato)]
        marcacoes = int(comprou)

        X.append(dados)
        Y.append(marcacoes)

    return X, Y