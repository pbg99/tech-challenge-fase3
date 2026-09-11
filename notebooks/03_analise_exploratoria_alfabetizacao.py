# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Análise Exploratória - Alfabetização Brasil
# MAGIC %md
# MAGIC # 📊 Análise Exploratória - Alfabetização no Brasil
# MAGIC
# MAGIC Análise descritiva e visual dos dados de alfabetização por UF, com foco em:
# MAGIC
# MAGIC * 📈 **Evolução temporal** das taxas de alfabetização
# MAGIC * 🗺️ **Comparação regional** e entre UFs
# MAGIC * 🏆 **Identificação de melhores e piores performances**
# MAGIC * 📉 **Análise de tendências** e volatilidade
# MAGIC * ⚖️ **Desigualdade** entre regiões e estados
# MAGIC * 💡 **Insights** e padrões relevantes
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Fonte de Dados:** Feature Store Silver (`workspace.silver.alfabetizacao_feature_store`)
# MAGIC
# MAGIC **Período Analisado:** Baseado nos dados disponíveis do SAEB/INEP
# MAGIC
# MAGIC **Cobertura:** 25 UFs brasileiras

# COMMAND ----------

# DBTITLE 1,Setup e Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F

# Configuração de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

print("✓ Bibliotecas carregadas")

# COMMAND ----------

# DBTITLE 1,1. Carregar Feature Store
# MAGIC %md
# MAGIC ## 1. 📥 Carregar Dados do Feature Store
# MAGIC
# MAGIC Carregando a tabela consolidada com todas as features derivadas

# COMMAND ----------

# DBTITLE 1,Carregar dados
# MAGIC %sql --name df_alfabetizacao
# MAGIC SELECT 
# MAGIC   ano,
# MAGIC   sigla_uf,
# MAGIC   nome_uf,
# MAGIC   regiao_geografica,
# MAGIC   rede,
# MAGIC   taxa_alfabetizacao,
# MAGIC   variacao_percentual,
# MAGIC   aceleracao,
# MAGIC   media_movel_3anos,
# MAGIC   tendencia,
# MAGIC   ranking_nacional,
# MAGIC   percentil_performance,
# MAGIC   quartil,
# MAGIC   classificacao_quartil,
# MAGIC   media_nacional,
# MAGIC   distancia_media_nacional,
# MAGIC   z_score,
# MAGIC   gap_desigualdade_nacional,
# MAGIC   indice_alfabetizacao_consolidado,
# MAGIC   score_progresso,
# MAGIC   volatilidade_3anos,
# MAGIC   classificacao_estabilidade,
# MAGIC   score_final_ponderado,
# MAGIC   media_regional,
# MAGIC   ranking_regional,
# MAGIC   distancia_media_regional,
# MAGIC   gap_desigualdade_regional
# MAGIC FROM workspace.silver.alfabetizacao_feature_store
# MAGIC ORDER BY ano DESC, rede, ranking_nacional;

# COMMAND ----------

# DBTITLE 1,Converter para Pandas
# Converter para Pandas para análises mais detalhadas
df = df_alfabetizacao.toPandas()

print(f"✓ Dados carregados: {len(df):,} registros")
print(f"✓ Período: {df['ano'].min()} - {df['ano'].max()}")
print(f"✓ UFs: {df['sigla_uf'].nunique()}")
print(f"✓ Redes: {', '.join(map(str, df['rede'].unique()))}")

# COMMAND ----------

# DBTITLE 1,2. Estatísticas Descritivas Gerais
# MAGIC %md
# MAGIC ## 2. 📊 Estatísticas Descritivas Gerais
# MAGIC
# MAGIC Visão geral dos dados de alfabetização

# COMMAND ----------

# DBTITLE 1,Estatísticas gerais
# Estatísticas descritivas da taxa de alfabetização
print("=" * 80)
print("ESTATÍSTICAS GERAIS - TAXA DE ALFABETIZAÇÃO (%)")
print("=" * 80)
print(df['taxa_alfabetizacao'].describe())
print()

# Por rede
print("\nPOR REDE:")
print(df.groupby('rede')['taxa_alfabetizacao'].describe().round(2))
print()

# Por região
print("\nPOR REGIÃO:")
print(df.groupby('regiao_geografica')['taxa_alfabetizacao'].describe().round(2))

# COMMAND ----------

# DBTITLE 1,Distribuição de Quartis
# Distribuição por quartil
print("\n" + "=" * 80)
print("DISTRIBUIÇÃO POR QUARTIL")
print("=" * 80)
quartil_dist = df.groupby(['rede', 'classificacao_quartil']).size().unstack(fill_value=0)
print(quartil_dist)
print(f"\nTotal de registros: {len(df)}")

# COMMAND ----------

# DBTITLE 1,3. Evolução Temporal
# MAGIC %md
# MAGIC ## 3. 📈 Evolução Temporal das Taxas de Alfabetização
# MAGIC
# MAGIC Análise da evolução ano a ano

# COMMAND ----------

# DBTITLE 1,Evolução por ano
# Evolução da média nacional por ano e rede
evolucao_ano = df.groupby(['ano', 'rede']).agg({
    'taxa_alfabetizacao': ['mean', 'std', 'min', 'max']
}).round(2)

print("=" * 80)
print("EVOLUÇÃO ANUAL - MÉDIA NACIONAL")
print("=" * 80)
print(evolucao_ano)

# Variação total no período
print("\n" + "=" * 80)
print("VARIAÇÃO TOTAL NO PERÍODO")
print("=" * 80)
for rede in df['rede'].unique():
    df_rede = df[df['rede'] == rede]
    taxa_inicial = df_rede[df_rede['ano'] == df_rede['ano'].min()]['taxa_alfabetizacao'].mean()
    taxa_final = df_rede[df_rede['ano'] == df_rede['ano'].max()]['taxa_alfabetizacao'].mean()
    variacao = taxa_final - taxa_inicial
    variacao_pct = (variacao / taxa_inicial) * 100 if taxa_inicial > 0 else 0
    print(f"\n{rede}:")
    print(f"  Inicial: {taxa_inicial:.2f}%")
    print(f"  Final: {taxa_final:.2f}%")
    print(f"  Variação: {variacao:+.2f} pontos percentuais ({variacao_pct:+.2f}%)")

# COMMAND ----------

# DBTITLE 1,Visualizar evolução temporal
# Gráfico de evolução temporal
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Por rede
for rede in df['rede'].unique():
    df_rede = df[df['rede'] == rede].groupby('ano')['taxa_alfabetizacao'].mean()
    axes[0].plot(df_rede.index, df_rede.values, marker='o', linewidth=2, label=rede)

axes[0].set_title('Evolução da Taxa de Alfabetização por Rede', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Ano', fontsize=12)
axes[0].set_ylabel('Taxa de Alfabetização (%)', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Por região
for regiao in df['regiao_geografica'].unique():
    df_regiao = df[df['regiao_geografica'] == regiao].groupby('ano')['taxa_alfabetizacao'].mean()
    axes[1].plot(df_regiao.index, df_regiao.values, marker='o', linewidth=2, label=regiao)

axes[1].set_title('Evolução da Taxa de Alfabetização por Região', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Ano', fontsize=12)
axes[1].set_ylabel('Taxa de Alfabetização (%)', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# DBTITLE 1,4. Análise Regional
# MAGIC %md
# MAGIC ## 4. 🗺️ Análise Regional
# MAGIC
# MAGIC Comparação entre as 5 regiões do Brasil

# COMMAND ----------

# DBTITLE 1,Comparação regional
# Dados mais recentes por região
ano_mais_recente = df['ano'].max()
df_recente = df[df['ano'] == ano_mais_recente]

print("=" * 80)
print(f"ANÁLISE REGIONAL - ANO {ano_mais_recente}")
print("=" * 80)

regional_stats = df_recente.groupby('regiao_geografica').agg({
    'taxa_alfabetizacao': ['mean', 'std', 'min', 'max'],
    'sigla_uf': 'count'
}).round(2)
regional_stats.columns = ['Média', 'Desvio Padrão', 'Mínima', 'Máxima', 'Nº UFs']
regional_stats = regional_stats.sort_values('Média', ascending=False)

print(regional_stats)
print(f"\nGap Nacional: {df_recente['gap_desigualdade_nacional'].iloc[0]:.2f} pontos percentuais")

# COMMAND ----------

# DBTITLE 1,Boxplot regional
# Boxplot por região
fig, ax = plt.subplots(figsize=(14, 6))

df_recente_sorted = df_recente.sort_values('taxa_alfabetizacao', ascending=False)
regiao_order = df_recente.groupby('regiao_geografica')['taxa_alfabetizacao'].median().sort_values(ascending=False).index

sns.boxplot(data=df_recente, x='regiao_geografica', y='taxa_alfabetizacao', 
            order=regiao_order, palette='Set2', ax=ax)
ax.set_title(f'Distribuição da Taxa de Alfabetização por Região ({ano_mais_recente})', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Região', fontsize=12)
ax.set_ylabel('Taxa de Alfabetização (%)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# DBTITLE 1,5. Top e Bottom Performers
# MAGIC %md
# MAGIC ## 5. 🏆 Top e Bottom Performers
# MAGIC
# MAGIC Melhores e piores UFs em alfabetização

# COMMAND ----------

# DBTITLE 1,Top e Bottom UFs
print("=" * 80)
print(f"TOP 10 UFs - MAIOR TAXA DE ALFABETIZAÇÃO ({ano_mais_recente})")
print("=" * 80)
top_10 = df_recente.nsmallest(10, 'ranking_nacional')[[
    'ranking_nacional', 'sigla_uf', 'nome_uf', 'regiao_geografica', 'rede',
    'taxa_alfabetizacao', 'score_final_ponderado'
]]
print(top_10.to_string(index=False))

print("\n" + "=" * 80)
print(f"BOTTOM 10 UFs - MENOR TAXA DE ALFABETIZAÇÃO ({ano_mais_recente})")
print("=" * 80)
bottom_10 = df_recente.nlargest(10, 'ranking_nacional')[[
    'ranking_nacional', 'sigla_uf', 'nome_uf', 'regiao_geografica', 'rede',
    'taxa_alfabetizacao', 'score_final_ponderado'
]]
print(bottom_10.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Ranking visual
# Top 15 e Bottom 15 UFs - visualização
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 15
top_15 = df_recente.nsmallest(15, 'ranking_nacional').sort_values('taxa_alfabetizacao')
axes[0].barh(range(len(top_15)), top_15['taxa_alfabetizacao'], color='green', alpha=0.7)
axes[0].set_yticks(range(len(top_15)))
axes[0].set_yticklabels([f"{row['sigla_uf']} - {row['rede']}" for _, row in top_15.iterrows()])
axes[0].set_xlabel('Taxa de Alfabetização (%)', fontsize=12)
axes[0].set_title('Top 15 UFs - Maior Taxa de Alfabetização', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Bottom 15
bottom_15 = df_recente.nlargest(15, 'ranking_nacional').sort_values('taxa_alfabetizacao', ascending=False)
axes[1].barh(range(len(bottom_15)), bottom_15['taxa_alfabetizacao'], color='red', alpha=0.7)
axes[1].set_yticks(range(len(bottom_15)))
axes[1].set_yticklabels([f"{row['sigla_uf']} - {row['rede']}" for _, row in bottom_15.iterrows()])
axes[1].set_xlabel('Taxa de Alfabetização (%)', fontsize=12)
axes[1].set_title('Bottom 15 UFs - Menor Taxa de Alfabetização', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# DBTITLE 1,6. Análise de Tendências
# MAGIC %md
# MAGIC ## 6. 📉 Análise de Tendências e Volatilidade
# MAGIC
# MAGIC Identificando UFs em crescimento, queda e estáveis

# COMMAND ----------

# DBTITLE 1,Distribuição de tendências
print("=" * 80)
print("DISTRIBUIÇÃO DE TENDÊNCIAS")
print("=" * 80)
tendencia_dist = df_recente.groupby(['tendencia', 'rede']).size().unstack(fill_value=0)
print(tendencia_dist)
print()

print("\n" + "=" * 80)
print("DISTRIBUIÇÃO DE VOLATILIDADE")
print("=" * 80)
volatilidade_dist = df_recente.groupby(['classificacao_estabilidade', 'rede']).size().unstack(fill_value=0)
print(volatilidade_dist)

# COMMAND ----------

# DBTITLE 1,UFs com maior crescimento
print("\n" + "=" * 80)
print("TOP 10 UFs - MAIOR CRESCIMENTO (Variação Percentual)")
print("=" * 80)
top_crescimento = df_recente.nlargest(10, 'variacao_percentual')[[
    'sigla_uf', 'nome_uf', 'regiao_geografica', 'rede',
    'taxa_alfabetizacao', 'variacao_percentual', 'tendencia'
]]
print(top_crescimento.to_string(index=False))

print("\n" + "=" * 80)
print("TOP 10 UFs - MAIOR QUEDA (Variação Percentual)")
print("=" * 80)
top_queda = df_recente.nsmallest(10, 'variacao_percentual')[[
    'sigla_uf', 'nome_uf', 'regiao_geografica', 'rede',
    'taxa_alfabetizacao', 'variacao_percentual', 'tendencia'
]]
print(top_queda.to_string(index=False))

# COMMAND ----------

# DBTITLE 1,7. Análise de Desigualdade
# MAGIC %md
# MAGIC ## 7. ⚖️ Análise de Desigualdade
# MAGIC
# MAGIC Gaps entre regiões e dentro de regiões

# COMMAND ----------

# DBTITLE 1,Gaps de desigualdade
print("=" * 80)
print("ANÁLISE DE DESIGUALDADE")
print("=" * 80)

# Gap nacional
gap_nacional = df_recente['gap_desigualdade_nacional'].iloc[0]
print(f"\nGap Nacional: {gap_nacional:.2f} pontos percentuais")
print(f"Taxa Mínima Nacional: {df_recente['taxa_alfabetizacao'].min():.2f}%")
print(f"Taxa Máxima Nacional: {df_recente['taxa_alfabetizacao'].max():.2f}%")

# Gap por região
print("\n" + "-" * 80)
print("GAP DE DESIGUALDADE POR REGIÃO")
print("-" * 80)
gap_regional = df_recente.groupby('regiao_geografica')['gap_desigualdade_regional'].first().sort_values(ascending=False)
print(gap_regional)

# Coeficiente de variação
print("\n" + "-" * 80)
print("COEFICIENTE DE VARIAÇÃO (CV) - Medida de Dispersão Relativa")
print("-" * 80)
cv_regional = (df_recente.groupby('regiao_geografica')['taxa_alfabetizacao'].std() / 
               df_recente.groupby('regiao_geografica')['taxa_alfabetizacao'].mean() * 100).sort_values(ascending=False)
print(cv_regional.round(2))

# COMMAND ----------

# DBTITLE 1,Visualizar desigualdade
# Mapa de calor - Score por UF e Região
fig, ax = plt.subplots(figsize=(14, 10))

# Preparar dados para heatmap
heatmap_data = df_recente.pivot_table(
    values='score_final_ponderado',
    index='sigla_uf',
    columns='rede',
    aggfunc='mean'
).sort_values(by=df_recente['rede'].unique()[0], ascending=False)

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
            cbar_kws={'label': 'Score Final Ponderado'}, ax=ax)
ax.set_title('Score Final Ponderado por UF e Rede', fontsize=14, fontweight='bold')
ax.set_xlabel('Rede', fontsize=12)
ax.set_ylabel('UF', fontsize=12)

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# DBTITLE 1,8. Insights e Conclusões
# MAGIC %md
# MAGIC ## 8. 💡 Principais Insights e Conclusões
# MAGIC
# MAGIC Síntese dos achados mais relevantes

# COMMAND ----------

# DBTITLE 1,Gerar insights automáticos
print("=" * 80)
print("PRINCIPAIS INSIGHTS - ALFABETIZAÇÃO NO BRASIL")
print("=" * 80)

# 1. Melhor e pior região
melhor_regiao = df_recente.groupby('regiao_geografica')['taxa_alfabetizacao'].mean().idxmax()
pior_regiao = df_recente.groupby('regiao_geografica')['taxa_alfabetizacao'].mean().idxmin()
print(f"\n1. DISPARIDADE REGIONAL:")
print(f"   ✓ Melhor região: {melhor_regiao} ({df_recente[df_recente['regiao_geografica']==melhor_regiao]['taxa_alfabetizacao'].mean():.2f}%)")
print(f"   ✓ Pior região: {pior_regiao} ({df_recente[df_recente['regiao_geografica']==pior_regiao]['taxa_alfabetizacao'].mean():.2f}%)")

# 2. UFs destaque
melhor_uf = df_recente.nsmallest(1, 'ranking_nacional').iloc[0]
pior_uf = df_recente.nlargest(1, 'ranking_nacional').iloc[0]
print(f"\n2. UFs DESTAQUE:")
print(f"   ✓ Melhor UF: {melhor_uf['sigla_uf']} - {melhor_uf['nome_uf']} ({melhor_uf['taxa_alfabetizacao']:.2f}%)")
print(f"   ✓ Pior UF: {pior_uf['sigla_uf']} - {pior_uf['nome_uf']} ({pior_uf['taxa_alfabetizacao']:.2f}%)")

# 3. Tendências
uf_crescente = df_recente[df_recente['tendencia'] == 'CRESCENTE'].shape[0]
uf_decrescente = df_recente[df_recente['tendencia'] == 'DECRESCENTE'].shape[0]
print(f"\n3. TENDÊNCIAS:")
print(f"   ✓ UFs em crescimento: {uf_crescente} ({uf_crescente/len(df_recente)*100:.1f}%)")
print(f"   ✓ UFs em queda: {uf_decrescente} ({uf_decrescente/len(df_recente)*100:.1f}%)")

# 4. Volatilidade
uf_estavel = df_recente[df_recente['classificacao_estabilidade'].isin(['MUITO ESTÁVEL', 'ESTÁVEL'])].shape[0]
print(f"\n4. ESTABILIDADE:")
print(f"   ✓ UFs estáveis: {uf_estavel} ({uf_estavel/len(df_recente)*100:.1f}%)")

# 5. Desigualdade
print(f"\n5. DESIGUALDADE:")
print(f"   ✓ Gap nacional: {gap_nacional:.2f} pontos percentuais")
print(f"   ✓ Região mais desigual: {gap_regional.idxmax()} ({gap_regional.max():.2f} pp)")
print(f"   ✓ Região mais homogênea: {gap_regional.idxmin()} ({gap_regional.min():.2f} pp)")

# 6. Score médio
score_medio = df_recente['score_final_ponderado'].mean()
print(f"\n6. PERFORMANCE GERAL:")
print(f"   ✓ Score final médio: {score_medio:.2f}/100")
print(f"   ✓ UFs acima da média: {(df_recente['score_final_ponderado'] > score_medio).sum()}")

print("\n" + "=" * 80)