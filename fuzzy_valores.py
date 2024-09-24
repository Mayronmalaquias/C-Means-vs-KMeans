import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import skfuzzy as fuzz  

input_file = "./2024-09-06.csv"

def remover_outliers_iqr(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    filtro = (df[coluna] >= (Q1 - 1.5 * IQR)) & (df[coluna] <= (Q3 + 1.5 * IQR))
    return df[filtro]

def calcular_rentabilidade(valor_locacao, valor_venda):
    return (valor_locacao * 12) / valor_venda if valor_venda != 0 else np.nan

def calcular_metricas_cluster(df, valor_coluna, oferta_tipo):
    metricas = {}

    if oferta_tipo == 'Venda':
        metricas['VALOR DE M² DE VENDA'] = df[valor_coluna].mean()
        metricas['VALOR DE VENDA NOMINAL'] = df['preco'].mean()
        metricas['METRAGEM MÉDIA DE VENDA'] = df['area_util'].mean()
        metricas['COEFICIENTE DE VARIAÇÃO DE VENDA'] = df[valor_coluna].std() / df[valor_coluna].mean() if df[valor_coluna].mean() != 0 else np.nan
        metricas['TAMANHO DA AMOSTRA DE VENDA'] = len(df)
    else:
        metricas['VALOR DE M² DE ALUGUEL'] = df[valor_coluna].mean()
        metricas['VALOR DE ALUGUEL NOMINAL'] = df['preco'].mean()
        metricas['METRAGEM MÉDIA DE ALUGUEL'] = df['area_util'].mean()
        metricas['COEFICIENTE DE VARIAÇÃO DE ALUGUEL'] = df[valor_coluna].std() / df[valor_coluna].mean() if df[valor_coluna].mean() != 0 else np.nan
        metricas['TAMANHO DA AMOSTRA DE ALUGUEL'] = len(df)

    return metricas

def formatar_resultados(df):
    df_formatted = df.copy()

    df_formatted['VALOR DE M² DE VENDA'] = df_formatted['VALOR DE M² DE VENDA'].apply(lambda x: f"R$ {x:,.2f} /m²")
    df_formatted['VALOR DE VENDA NOMINAL'] = df_formatted['VALOR DE VENDA NOMINAL'].apply(lambda x: f"R$ {x:,.2f}")
    df_formatted['METRAGEM MÉDIA DE VENDA'] = df_formatted['METRAGEM MÉDIA DE VENDA'].apply(lambda x: f"{x:.2f} m²")

    df_formatted['VALOR DE M² DE ALUGUEL'] = df_formatted['VALOR DE M² DE ALUGUEL'].apply(lambda x: f"R$ {x:,.2f} /m²")
    df_formatted['VALOR DE ALUGUEL NOMINAL'] = df_formatted['VALOR DE ALUGUEL NOMINAL'].apply(lambda x: f"R$ {x:,.2f}")
    df_formatted['METRAGEM MÉDIA DE ALUGUEL'] = df_formatted['METRAGEM MÉDIA DE ALUGUEL'].apply(lambda x: f"{x:.2f} m²")

    df_formatted['COEFICIENTE DE VARIAÇÃO DE VENDA'] = df_formatted['COEFICIENTE DE VARIAÇÃO DE VENDA'].apply(lambda x: f"{x:.2%}")
    df_formatted['COEFICIENTE DE VARIAÇÃO DE ALUGUEL'] = df_formatted['COEFICIENTE DE VARIAÇÃO DE ALUGUEL'].apply(lambda x: f"{x:.2%}")

    df_formatted['RENTABILIDADE MÉDIA'] = df_formatted['RENTABILIDADE MÉDIA'].apply(lambda x: f"{x:.2%}")

    return df_formatted

def clusterizar_kmeans(df, valor_coluna, oferta_tipo, n_clusters=9):
    df_oferta = df[df['oferta'] == oferta_tipo].copy()

    if not df_oferta.empty and valor_coluna in df_oferta.columns and len(df_oferta) >= n_clusters:
        X = df_oferta[[valor_coluna]].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_oferta.loc[:, "cluster"] = kmeans.fit_predict(X)
        df_oferta.loc[:, 'cluster'] = df_oferta['cluster'].astype('category')

        metricas_clusters = []
        for cluster in sorted(df_oferta['cluster'].unique()):
            cluster_data = df_oferta[df_oferta['cluster'] == cluster]
            metricas = calcular_metricas_cluster(cluster_data, valor_coluna, oferta_tipo)
            metricas['Cluster'] = cluster
            metricas_clusters.append(metricas)

        metricas_df = pd.DataFrame(metricas_clusters)

        if oferta_tipo == 'Venda':
            metricas_df = metricas_df.reindex(columns=['VALOR DE M² DE VENDA', 'VALOR DE VENDA NOMINAL',
                                                       'METRAGEM MÉDIA DE VENDA', 'COEFICIENTE DE VARIAÇÃO DE VENDA',
                                                       'TAMANHO DA AMOSTRA DE VENDA'], fill_value=np.nan)
        else:
            metricas_df = metricas_df.reindex(columns=['VALOR DE M² DE ALUGUEL', 'VALOR DE ALUGUEL NOMINAL',
                                                       'METRAGEM MÉDIA DE ALUGUEL', 'COEFICIENTE DE VARIAÇÃO DE ALUGUEL',
                                                       'TAMANHO DA AMOSTRA DE ALUGUEL'], fill_value=np.nan)
        return metricas_df.reset_index(drop=True)
    else:
        return pd.DataFrame()

def clusterizar_fuzzy(df, valor_coluna, oferta_tipo, n_clusters=9):
    df_oferta = df[df['oferta'] == oferta_tipo].copy()

    if not df_oferta.empty and valor_coluna in df_oferta.columns and len(df_oferta) >= n_clusters:
        X = df_oferta[[valor_coluna]].values.T  # Necessário transpor para Fuzzy C-Means
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X, n_clusters, 2, error=0.005, maxiter=1000, seed=42)

        df_oferta['cluster'] = np.argmax(u, axis=0)  # Removemos .astype('category')

        metricas_clusters = []
        for cluster in sorted(df_oferta['cluster'].unique()):
            cluster_data = df_oferta[df_oferta['cluster'] == cluster]
            metricas = calcular_metricas_cluster(cluster_data, valor_coluna, oferta_tipo)
            metricas['Cluster'] = cluster
            metricas_clusters.append(metricas)

        metricas_df = pd.DataFrame(metricas_clusters)

        if oferta_tipo == 'Venda':
            metricas_df = metricas_df.reindex(columns=['VALOR DE M² DE VENDA', 'VALOR DE VENDA NOMINAL',
                                                       'METRAGEM MÉDIA DE VENDA', 'COEFICIENTE DE VARIAÇÃO DE VENDA',
                                                       'TAMANHO DA AMOSTRA DE VENDA'], fill_value=np.nan)
        else:
            metricas_df = metricas_df.reindex(columns=['VALOR DE M² DE ALUGUEL', 'VALOR DE ALUGUEL NOMINAL',
                                                       'METRAGEM MÉDIA DE ALUGUEL', 'COEFICIENTE DE VARIAÇÃO DE ALUGUEL',
                                                       'TAMANHO DA AMOSTRA DE ALUGUEL'], fill_value=np.nan)
        return metricas_df.reset_index(drop=True)
    else:
        return pd.DataFrame()


def comparar_resultados(tipo_imovel=None, bairro=None):
    df = pd.read_csv(input_file, sep=",", thousands=".", decimal=",")
    
    filtro = pd.Series([True] * len(df))

    if tipo_imovel:
        filtro &= (df["tipo"] == tipo_imovel)
    if bairro:
        filtro &= (df["bairro"] == bairro)

    df_filtrado = df[filtro]

    metricas_kmeans = clusterizar_kmeans(df_filtrado, "valor_m2", "Venda")
    
    metricas_fuzzy = clusterizar_fuzzy(df_filtrado, "valor_m2", "Venda")
    
    print("\nResultados KMeans:\n", metricas_kmeans)

    print("\nResultados Fuzzy C-Means:\n", metricas_fuzzy)

    return metricas_kmeans, metricas_fuzzy

def analisar_imovel_detalhado(tipo_imovel=None, bairro=None, cidade=None, cep=None, vaga_garagem=None, quadra=None, quartos=None, metragem=None):
    df = pd.read_csv(input_file, sep=",", thousands=".", decimal=",")

    print(f"Total de registros no dataset: {len(df)}")

    filtro = pd.Series([True] * len(df))

    if tipo_imovel:
        filtro &= (df["tipo"] == tipo_imovel)
    if bairro:
        filtro &= (df["bairro"] == bairro)
    if cidade:
        filtro &= (df["cidade"] == cidade)
    if cep:
        filtro &= (df["cep"] == cep)
    if vaga_garagem is not None:
        filtro &= (df["vagas"].notnull() if vaga_garagem else df["vagas"].isnull())
    if quadra:
        filtro &= (df["quadra"] == quadra)
    if quartos:
        filtro &= (df["quartos"] == quartos)
    if metragem:
        filtro &= ((df["area_util"] >= metragem * 0.9) & (df["area_util"] <= metragem * 1.1))

    df_filtrado = df[filtro]

    print(f"Total de registros após filtragem: {len(df_filtrado)}")

    comparar_resultados(tipo_imovel=tipo_imovel, bairro=bairro)

    print("\nComparação entre KMeans e Fuzzy C-Means concluída!")

print(analisar_imovel_detalhado(tipo_imovel="Apartamento", bairro="ASA SUL", cidade=None, cep=None, vaga_garagem=None, quadra=None, quartos=None, metragem=None))

