# Projeto Acadêmico. Tema: Previsão de Qualidade de Cervejas com váriaveis físico-químicas.
# Consulte o README para mais detalhes.

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

ARQUIVO_TREINO = "cerveja_treino.csv"
ARQUIVO_TESTE = "cerveja_teste.csv"


def carregar_csv(arquivo):
    try:
        df = pd.read_csv(arquivo, decimal=",", header=None)
        return df.values
    except Exception:
        data = []
        with open(arquivo, "r", encoding="utf-8") as f:
            for line in f:
                parts = []
                current = ""
                in_quotes = False

                for char in line.strip():
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == "," and not in_quotes:
                        parts.append(current.strip())
                        current = ""
                    else:
                        current += char
                parts.append(current.strip())

                row = []
                for part in parts:
                    part = part.replace('"', "")
                    if "," in part:
                        row.append(float(part.replace(",", ".")))
                    else:
                        row.append(float(part))

                if len(row) in [7, 8]:  # 7 para teste, 8 para treino
                    data.append(row)

        return np.array(data)


def mse_loss(params, X_subset, y_target):
    # Mean Squared Error
    X_with_bias = np.column_stack([np.ones(len(X_subset)), X_subset])
    y_pred = X_with_bias @ params
    return np.mean((y_target - y_pred) ** 2)


def mse_gradient(params, X_subset, y_target):
    # Gradiente do MSE
    X_with_bias = np.column_stack([np.ones(len(X_subset)), X_subset])
    y_pred = X_with_bias @ params
    return 2 * X_with_bias.T @ (y_pred - y_target) / len(y_target)


def scipy_otimizar(X_subset, y_target):
    # Otimização com scipy.optimize.minimize (método BFGS)
    X_with_bias = np.column_stack([np.ones(len(X_subset)), X_subset])
    params_init = np.random.normal(0, 0.01, X_with_bias.shape[1])

    resultado = minimize(
        fun=mse_loss,
        x0=params_init,
        args=(X_subset, y_target),
        jac=mse_gradient,
        method="BFGS",
    )

    return resultado.x, resultado.fun


def gradiente_proprio(X_subset, y_target, lr=0.01, max_iter=1000):
    # Gradiente descendente implementado do zero
    X_with_bias = np.column_stack([np.ones(len(X_subset)), X_subset])
    params = np.random.normal(0, 0.01, X_with_bias.shape[1])

    for _ in range(max_iter):
        grad = mse_gradient(params, X_subset, y_target)
        params_new = params - lr * grad

        if np.linalg.norm(params_new - params) < 1e-6:
            break
        params = params_new

    score = mse_loss(params, X_subset, y_target)
    return params, score


def executar_projeto_completo():

    print("PROJETO MÍNIMOS QUADRADOS - DADOS DE CERVEJA")
    print("=" * 60)

    print("Carregando dados de treino...")
    data_treino = carregar_csv(ARQUIVO_TREINO)
    X_treino = data_treino[:, :-1]  # Primeiras 7 colunas
    y_treino = data_treino[:, -1]  # Última coluna

    print(f" Treino: {X_treino.shape[0]} amostras × {X_treino.shape[1]} features")

    print("Carregando dados de teste...")
    X_teste = carregar_csv(ARQUIVO_TESTE)

    print(f" Teste: {X_teste.shape[0]} amostras × {X_teste.shape[1]} features")

    print("\nNormalizando dados...")

    X_mean = np.mean(X_treino, axis=0)
    X_std = np.std(X_treino, axis=0)
    X_treino_norm = (X_treino - X_mean) / X_std

    y_mean = np.mean(y_treino)
    y_std = np.std(y_treino)
    y_treino_norm = (y_treino - y_mean) / y_std

    print(" Dados normalizados (média≈0, std≈1)")

    print("\nBuscando melhores combinações de variáveis...")

    melhores_modelos = {}

    for R in range(1, 5):  
        print(f"\n--- R = {R} variáveis ---")

        melhor_score = float("inf")
        melhor_modelo = None

        for combinacao in combinations(range(7), R):
            X_subset = X_treino_norm[:, list(combinacao)]

            try:
                params_scipy, score_scipy = scipy_otimizar(X_subset, y_treino_norm)
                params_grad, score_grad = gradiente_proprio(X_subset, y_treino_norm)

                if score_scipy < melhor_score:
                    melhor_score = score_scipy
                    melhor_modelo = {
                        "combinacao": combinacao,
                        "features": [f"X{i}" for i in combinacao],
                        "params_scipy": params_scipy,
                        "params_grad": params_grad,
                        "score_scipy": score_scipy,
                        "score_grad": score_grad,
                    }
            except Exception:
                continue

        if melhor_modelo:
            melhores_modelos[R] = melhor_modelo
            print(f" Melhor: {melhor_modelo['features']}")
            print(f"  MSE scipy: {melhor_modelo['score_scipy']:.6f}")
            print(f"  MSE gradiente: {melhor_modelo['score_grad']:.6f}")

    print("\nRELATÓRIO DE TREINAMENTO")
    print("=" * 40)

    for R in sorted(melhores_modelos.keys()):
        modelo = melhores_modelos[R]

        X_subset = X_treino_norm[:, list(modelo["combinacao"])]
        X_with_bias = np.column_stack([np.ones(len(X_subset)), X_subset])
        y_pred_norm = X_with_bias @ modelo["params_scipy"]
        y_pred = y_pred_norm * y_std + y_mean

        ss_res = np.sum((y_treino - y_pred) ** 2)
        ss_tot = np.sum((y_treino - np.mean(y_treino)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        modelo["r2"] = r2

        print(f"\nR={R}: {modelo['features']}")
        print(f"  MSE (normalizado): {modelo['score_scipy']:.6f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Diferença scipy/gradiente: {abs(modelo['score_scipy'] - modelo['score_grad']):.6f}")

    print("\nAPLICANDO MODELOS EM DADOS DE TESTE")
    print("=" * 50)

    X_teste_norm = (X_teste - X_mean) / X_std

    previsoes_finais = {}

    for R in melhores_modelos:
        modelo = melhores_modelos[R]
        indices = list(modelo["combinacao"])

        print(f"\nModelo R={R}: {modelo['features']}")

        X_subset = X_teste_norm[:, indices]
        X_with_bias = np.column_stack([np.ones(len(X_subset)), X_subset])

        y_pred_norm_scipy = X_with_bias @ modelo["params_scipy"]
        y_pred_scipy = y_pred_norm_scipy * y_std + y_mean

        y_pred_norm_grad = X_with_bias @ modelo["params_grad"]
        y_pred_grad = y_pred_norm_grad * y_std + y_mean

        previsoes_finais[R] = {"scipy": y_pred_scipy, "gradiente": y_pred_grad}

        print(f"  Previsões geradas: [{y_pred_scipy.min():.1f}, {y_pred_scipy.max():.1f}]")
        print(f"  Diferença scipy/gradiente: {np.max(np.abs(y_pred_scipy - y_pred_grad)):.3f}")

    print("\nSalvando arquivos de resultado...")

    for R in previsoes_finais:
        np.savetxt(f"previsoes_R{R}_scipy.txt", previsoes_finais[R]["scipy"], fmt="%.6f")
        np.savetxt(f"previsoes_R{R}_gradiente.txt", previsoes_finais[R]["gradiente"], fmt="%.6f")

        print(f"  previsoes_R{R}_scipy.txt e previsoes_R{R}_gradiente.txt")

    with open("relatorio_final.txt", "w", encoding="utf-8") as f:
        f.write("PROJETO MÍNIMOS QUADRADOS - RELATÓRIO FINAL\n")
        f.write("=" * 50 + "\n\n")

        f.write("O QUE É ESTE ARQUIVO?\n")
        f.write("-" * 50 + "\n")
        f.write(
            "Este relatório resume os resultados do experimento: para cada\n"
            "quantidade de variáveis testada (R), mostra qual foi a melhor\n"
            "combinação encontrada, o quão bem o modelo se ajustou aos dados\n"
            "de treino, e as previsões geradas para os dados de teste.\n\n"
        )

        f.write("GLOSSÁRIO RÁPIDO\n")
        f.write("-" * 50 + "\n")
        f.write(
            "R           = número de variáveis (features) usadas no modelo.\n"
            "              O script testa todas as combinações possíveis de\n"
            "              1 a 4 variáveis dentre as 7 disponíveis, e guarda\n"
            "              aqui apenas a melhor combinação de cada tamanho.\n"
            "X0, X1, ...   = nomes genéricos das colunas dos dados de entrada\n"
            "              (X0 é a 1ª coluna, X1 a 2ª, e assim por diante).\n"
            "MSE         = Erro Quadrático Médio. Mede o quão distante as\n"
            "              previsões do modelo estão dos valores reais.\n"
            "              QUANTO MENOR, MELHOR o ajuste do modelo.\n"
            "R²          = mede a proporção da variação dos dados que o\n"
            "              modelo consegue explicar. Vai de 0 a 1: quanto\n"
            "              mais perto de 1, melhor o modelo se ajusta aos\n"
            "              dados de treino.\n"
            "Scipy vs.     o projeto calcula os coeficientes do modelo de\n"
            "Gradiente     duas formas diferentes (uma usando a biblioteca\n"
            "              SciPy, outra com gradiente descendente escrito do\n"
            "              zero). Se os dois métodos derem resultados bem\n"
            "              parecidos, isso é um bom sinal de que a\n"
            "              implementação própria está correta.\n\n"
        )

        f.write("DADOS UTILIZADOS\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Treino: {X_treino.shape[0]} amostras × {X_treino.shape[1]} variáveis (features)\n")
        f.write(f"  Teste:  {X_teste.shape[0]} amostras × {X_teste.shape[1]} variáveis (features)\n\n")

        f.write("MELHORES MODELOS ENCONTRADOS (por quantidade de variáveis)\n")
        f.write("-" * 50 + "\n")
        for R in sorted(melhores_modelos.keys()):
            modelo = melhores_modelos[R]
            diferenca = abs(modelo["score_scipy"] - modelo["score_grad"])
            f.write(f"\n  R={R} variável(is): {modelo['features']}\n")
            f.write(f"    MSE (scipy):      {modelo['score_scipy']:.6f}  <- quanto menor, melhor\n")
            f.write(f"    MSE (gradiente):  {modelo['score_grad']:.6f}\n")
            f.write(f"    R² (treino):      {modelo['r2']:.4f}  <- quanto mais perto de 1, melhor\n")
            f.write(f"    Diferença scipy/gradiente: {diferenca:.6f}  <- quanto menor, mais os dois métodos concordam\n")

        f.write("\n\nPREVISÕES GERADAS PARA OS DADOS DE TESTE\n")
        f.write("-" * 50 + "\n")
        f.write(
            "Faixa de valores (mínimo, máximo e média) prevista por cada\n"
            "modelo para os dados de teste. Útil para checar se as\n"
            "previsões fazem sentido (por exemplo, sem valores absurdos).\n"
        )
        for R in previsoes_finais:
            pred_scipy = previsoes_finais[R]["scipy"]
            pred_grad = previsoes_finais[R]["gradiente"]
            f.write(f"\n  R={R}:\n")
            f.write(f"    Scipy:     min={pred_scipy.min():.1f}, max={pred_scipy.max():.1f}, média={pred_scipy.mean():.1f}\n")
            f.write(f"    Gradiente: min={pred_grad.min():.1f}, max={pred_grad.max():.1f}, média={pred_grad.mean():.1f}\n")

        f.write("\n\nCOMO ESCOLHER O MELHOR MODELO?\n")
        f.write("-" * 50 + "\n")
        f.write(
            "Regra geral: prefira o modelo com menor MSE e maior R² no\n"
            "treino. Mas fique atento(a) — usar mais variáveis quase\n"
            "sempre reduz o MSE de treino, o que pode ser só overfitting\n"
            "(o modelo 'decorou' os dados de treino, em vez de aprender um\n"
            "padrão que generaliza). Por isso, vale comparar também se o\n"
            "ganho de MSE entre um R e o próximo é significativo, ou\n"
            "apenas marginal.\n"
        )

    print(" relatorio_final.txt")
    print("\n" + "=" * 60)
    print("CONCLUÍDO COM SUCESSO!")
    print("=" * 60)

    print("\nArquivos gerados:")
    for R in range(1, 5):
        if R in previsoes_finais:
            print(f" previsoes_R{R}_scipy.txt")
            print(f"  previsoes_R{R}_gradiente.txt")
    print("  relatorio_final.txt")

    print("\nResumo dos melhores modelos:")
    for R in sorted(melhores_modelos.keys()):
        modelo = melhores_modelos[R]
        print(f"  R={R}: {modelo['features']} (MSE: {modelo['score_scipy']:.6f})")

    return melhores_modelos, previsoes_finais


if __name__ == "__main__":
    modelos, previsoes = executar_projeto_completo()
    print("\nTodos os arquivos necessários foram gerados.")