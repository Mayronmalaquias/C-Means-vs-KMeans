import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import seaborn as sns

input_file = "./2024-09-06.csv"

def remover_outliers_iqr(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    filtro = (df[coluna] >= (Q1 - 1.5 * IQR)) & (df[coluna] <= (Q3 + 1.5 * IQR))
    return df[filtro]

def clusterizar_kmeans(df, valor_coluna, oferta_tipo, n_clusters=9):
    df_oferta = df[df['oferta'] == oferta_tipo].copy()
    if not df_oferta.empty and valor_coluna in df_oferta.columns and len(df_oferta) >= n_clusters:
        X = df_oferta[[valor_coluna]].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_oferta["cluster"] = kmeans.fit_predict(X)
        return df_oferta
    else:
        return pd.DataFrame()

def clusterizar_fuzzy(df, valor_coluna, oferta_tipo, n_clusters=9):
    df_oferta = df[df['oferta'] == oferta_tipo].copy()
    if not df_oferta.empty and valor_coluna in df_oferta.columns and len(df_oferta) >= n_clusters:
        X = df_oferta[[valor_coluna]].values.T  # Transpor para Fuzzy C-Means
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X, n_clusters, 2, error=0.005, maxiter=1000, seed=42)
        df_oferta["cluster"] = np.argmax(u, axis=0)
        return df_oferta
    else:
        return pd.DataFrame()

def plot_cluster_comparison(df_kmeans, df_fuzzy, valor_coluna, cluster_num=0):
    plt.figure(figsize=(16, 10))

    df_kmeans_cluster = df_kmeans[df_kmeans['cluster'] == cluster_num]
    df_fuzzy_cluster = df_fuzzy[df_fuzzy['cluster'] == cluster_num]

    plt.subplot(2, 2, 1)
    plt.scatter(df_kmeans_cluster[valor_coluna], df_kmeans_cluster['area_util'], 
                c='blue', s=df_kmeans_cluster['preco'] / 20000, alpha=0.6, label=f'KMeans - Cluster {cluster_num}', edgecolors='w')
    plt.title(f'KMeans - Cluster {cluster_num}')
    plt.xlabel('Valor por m² (Venda)')
    plt.ylabel('Área Útil')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.scatter(df_fuzzy_cluster[valor_coluna], df_fuzzy_cluster['area_util'], 
                c='green', s=df_fuzzy_cluster['preco'] / 20000, alpha=0.6, label=f'Fuzzy C-Means - Cluster {cluster_num}', edgecolors='w')
    plt.title(f'Fuzzy C-Means - Cluster {cluster_num}')
    plt.xlabel('Valor por m² (Venda)')
    plt.ylabel('Área Útil')
    plt.legend()

    plt.subplot(2, 2, 3)
    kmeans_cluster_count = df_kmeans_cluster.shape[0]
    fuzzy_cluster_count = df_fuzzy_cluster.shape[0]

    labels = ['KMeans', 'Fuzzy C-Means']
    counts = [kmeans_cluster_count, fuzzy_cluster_count]

    plt.bar(labels, counts, color=['blue', 'green'])
    plt.xlabel('Método de Clusterização')
    plt.ylabel('Quantidade de Imóveis')
    plt.title(f'Quantidade de Imóveis no Cluster {cluster_num}')

    plt.subplot(2, 2, 4)
    sns.kdeplot(df_kmeans_cluster[valor_coluna], color='blue', label=f'KMeans - Cluster {cluster_num}', fill=True)
    sns.kdeplot(df_fuzzy_cluster[valor_coluna], color='green', label=f'Fuzzy C-Means - Cluster {cluster_num}', fill=True)
    plt.title(f'Densidade de Valor por m² - Cluster {cluster_num}')
    plt.xlabel('Valor por m² (Venda)')
    plt.ylabel('Densidade')
    plt.legend()

    plt.tight_layout()
    plt.show()

def comparar_cluster(tipo_imovel=None, bairro=None, cluster_num=0):
    df = pd.read_csv(input_file, sep=",", thousands=".", decimal=",")

    filtro = pd.Series([True] * len(df))
    if tipo_imovel:
        filtro &= (df["tipo"] == tipo_imovel)
    if bairro:
        filtro &= (df["bairro"] == bairro)

    df_filtrado = df[filtro]

    df_kmeans = clusterizar_kmeans(df_filtrado, "valor_m2", "Venda")
    df_fuzzy = clusterizar_fuzzy(df_filtrado, "valor_m2", "Venda")

    plot_cluster_comparison(df_kmeans, df_fuzzy, "valor_m2", cluster_num)

    return df_kmeans, df_fuzzy

def analisar_cluster_especifico(tipo_imovel=None, bairro=None, cluster_num=0):
    print(f"Comparando o Cluster {cluster_num} entre KMeans e Fuzzy C-Means")
    comparar_cluster(tipo_imovel=tipo_imovel, bairro=bairro, cluster_num=cluster_num)

analisar_cluster_especifico(tipo_imovel="Apartamento", bairro="ASA SUL", cluster_num=5)
