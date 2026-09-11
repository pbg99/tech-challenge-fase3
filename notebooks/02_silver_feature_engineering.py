# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Introdução - Engenharia de Atributos
# MAGIC %md
# MAGIC # 02 — Silver: Engenharia de Atributos (Feature Engineering)
# MAGIC
# MAGIC Camada Silver: transforma dados brutos do Bronze em features analíticas derivadas.
# MAGIC
# MAGIC ## Atributos Derivados:
# MAGIC
# MAGIC ### 📊 **Evolução Temporal**
# MAGIC * Variação ano a ano (delta e percentual)
# MAGIC * Média móvel (últimos 3 anos)
# MAGIC * Tendência linear
# MAGIC * Aceleração (variação da variação)
# MAGIC
# MAGIC ### 🎯 **Indicadores de Performance**
# MAGIC * Distância para meta nacional/estadual
# MAGIC * Ranking entre UFs
# MAGIC * Classificação por quartis
# MAGIC * Gap de desigualdade (diferença entre maior e menor por ano)
# MAGIC
# MAGIC ### 🔄 **Indicadores Compostos**
# MAGIC * Índice de Alfabetização Consolidado (média ponderada por nível)
# MAGIC * Score de Progresso (evolução + posição atual)
# MAGIC * Volatilidade (desvio padrão últimos 3 anos)
# MAGIC
# MAGIC ### 🌐 **Enriquecimento Contextual**
# MAGIC * Join com dados PNAD (contexto socioeconômico)
# MAGIC * Join com dados FUNDEB (investimento em educação)
# MAGIC * Densidade populacional (UF)
# MAGIC * Região geográfica

# COMMAND ----------

# DBTITLE 1,Setup - Variáveis
# MAGIC %sql
# MAGIC -- Configuração
# MAGIC DECLARE OR REPLACE VARIABLE CATALOG STRING DEFAULT 'workspace';
# MAGIC DECLARE OR REPLACE VARIABLE ANO_BASE INT DEFAULT 2023; -- Ano mais recente disponível

# COMMAND ----------

# DBTITLE 1,1. Evolução Temporal - Base
# MAGIC %md
# MAGIC ## 1. Evolução Temporal - Métricas por UF e Ano
# MAGIC
# MAGIC Calcular variações ano a ano, tendências e médias móveis

# COMMAND ----------

# DBTITLE 1,Criar tabela de evolução temporal
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.silver.alfabetizacao_evolucao_temporal
# MAGIC USING DELTA
# MAGIC AS
# MAGIC WITH base AS (
# MAGIC   SELECT 
# MAGIC     ano,
# MAGIC     sigla_uf,
# MAGIC     rede,
# MAGIC     taxa_alfabetizacao,
# MAGIC     -- Lag para variação ano a ano
# MAGIC     LAG(taxa_alfabetizacao, 1) OVER (PARTITION BY sigla_uf, rede ORDER BY ano) AS taxa_ano_anterior,
# MAGIC     LAG(taxa_alfabetizacao, 2) OVER (PARTITION BY sigla_uf, rede ORDER BY ano) AS taxa_2_anos_atras,
# MAGIC     -- Lead para projeção
# MAGIC     LEAD(taxa_alfabetizacao, 1) OVER (PARTITION BY sigla_uf, rede ORDER BY ano) AS taxa_ano_seguinte
# MAGIC   FROM workspace.bronze.avaliacao_alfabetizacao
# MAGIC   WHERE taxa_alfabetizacao IS NOT NULL
# MAGIC )
# MAGIC SELECT 
# MAGIC   ano,
# MAGIC   sigla_uf,
# MAGIC   rede,
# MAGIC   taxa_alfabetizacao,
# MAGIC   
# MAGIC   -- 1. Variação Absoluta e Percentual
# MAGIC   taxa_alfabetizacao - taxa_ano_anterior AS variacao_absoluta,
# MAGIC   ROUND(
# MAGIC     CASE 
# MAGIC       WHEN taxa_ano_anterior > 0 
# MAGIC       THEN ((taxa_alfabetizacao - taxa_ano_anterior) / taxa_ano_anterior) * 100
# MAGIC       ELSE NULL 
# MAGIC     END, 
# MAGIC     2
# MAGIC   ) AS variacao_percentual,
# MAGIC   
# MAGIC   -- 2. Aceleração (variação da variação)
# MAGIC   ROUND(
# MAGIC     (taxa_alfabetizacao - taxa_ano_anterior) - (taxa_ano_anterior - taxa_2_anos_atras),
# MAGIC     2
# MAGIC   ) AS aceleracao,
# MAGIC   
# MAGIC   -- 3. Média Móvel (últimos 3 anos)
# MAGIC   ROUND(
# MAGIC     AVG(taxa_alfabetizacao) OVER (
# MAGIC       PARTITION BY sigla_uf, rede 
# MAGIC       ORDER BY ano 
# MAGIC       ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
# MAGIC     ),
# MAGIC     2
# MAGIC   ) AS media_movel_3anos,
# MAGIC   
# MAGIC   -- 4. Tendência (diferença entre atual e média móvel)
# MAGIC   ROUND(
# MAGIC     taxa_alfabetizacao - AVG(taxa_alfabetizacao) OVER (
# MAGIC       PARTITION BY sigla_uf, rede 
# MAGIC       ORDER BY ano 
# MAGIC       ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
# MAGIC     ),
# MAGIC     2
# MAGIC   ) AS desvio_media_movel,
# MAGIC   
# MAGIC   -- 5. Flag de tendência
# MAGIC   CASE 
# MAGIC     WHEN taxa_alfabetizacao > taxa_ano_anterior THEN 'CRESCENTE'
# MAGIC     WHEN taxa_alfabetizacao < taxa_ano_anterior THEN 'DECRESCENTE'
# MAGIC     ELSE 'ESTÁVEL'
# MAGIC   END AS tendencia
# MAGIC   
# MAGIC FROM base
# MAGIC ORDER BY sigla_uf, rede, ano;
# MAGIC
# MAGIC SELECT '✓ Evolução temporal criada: ' || COUNT(*) || ' registros' AS status
# MAGIC FROM workspace.silver.alfabetizacao_evolucao_temporal;

# COMMAND ----------

# DBTITLE 1,2. Indicadores de Performance
# MAGIC %md
# MAGIC ## 2. Indicadores de Performance - Rankings e Metas
# MAGIC
# MAGIC Ranking entre UFs, distância para meta, classificação por quartis

# COMMAND ----------

# DBTITLE 1,Criar tabela de performance
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.silver.alfabetizacao_performance
# MAGIC USING DELTA
# MAGIC AS
# MAGIC WITH base AS (
# MAGIC   SELECT 
# MAGIC     ano,
# MAGIC     sigla_uf,
# MAGIC     rede,
# MAGIC     taxa_alfabetizacao
# MAGIC   FROM workspace.bronze.avaliacao_alfabetizacao
# MAGIC   WHERE taxa_alfabetizacao IS NOT NULL
# MAGIC ),
# MAGIC ranking AS (
# MAGIC   SELECT 
# MAGIC     *,
# MAGIC     -- Ranking nacional por ano e rede
# MAGIC     RANK() OVER (PARTITION BY ano, rede ORDER BY taxa_alfabetizacao DESC) AS ranking_nacional,
# MAGIC     DENSE_RANK() OVER (PARTITION BY ano, rede ORDER BY taxa_alfabetizacao DESC) AS ranking_nacional_denso,
# MAGIC     
# MAGIC     -- Percentil
# MAGIC     PERCENT_RANK() OVER (PARTITION BY ano, rede ORDER BY taxa_alfabetizacao) AS percentil,
# MAGIC     
# MAGIC     -- Quartil
# MAGIC     NTILE(4) OVER (PARTITION BY ano, rede ORDER BY taxa_alfabetizacao) AS quartil,
# MAGIC     
# MAGIC     -- Estatísticas para contexto
# MAGIC     AVG(taxa_alfabetizacao) OVER (PARTITION BY ano, rede) AS media_nacional,
# MAGIC     STDDEV(taxa_alfabetizacao) OVER (PARTITION BY ano, rede) AS desvio_padrao_nacional,
# MAGIC     MIN(taxa_alfabetizacao) OVER (PARTITION BY ano, rede) AS taxa_minima_nacional,
# MAGIC     MAX(taxa_alfabetizacao) OVER (PARTITION BY ano, rede) AS taxa_maxima_nacional
# MAGIC   FROM base
# MAGIC )
# MAGIC SELECT 
# MAGIC   ano,
# MAGIC   sigla_uf,
# MAGIC   rede,
# MAGIC   taxa_alfabetizacao,
# MAGIC   
# MAGIC   -- Rankings
# MAGIC   ranking_nacional,
# MAGIC   ranking_nacional_denso,
# MAGIC   ROUND(percentil * 100, 2) AS percentil_performance,
# MAGIC   quartil,
# MAGIC   
# MAGIC   -- Classificação por quartil
# MAGIC   CASE 
# MAGIC     WHEN quartil = 1 THEN 'BAIXO (Q1)'
# MAGIC     WHEN quartil = 2 THEN 'MÉDIO-BAIXO (Q2)'
# MAGIC     WHEN quartil = 3 THEN 'MÉDIO-ALTO (Q3)'
# MAGIC     WHEN quartil = 4 THEN 'ALTO (Q4)'
# MAGIC   END AS classificacao_quartil,
# MAGIC   
# MAGIC   -- Comparação com média nacional
# MAGIC   ROUND(media_nacional, 2) AS media_nacional,
# MAGIC   ROUND(taxa_alfabetizacao - media_nacional, 2) AS distancia_media_nacional,
# MAGIC   
# MAGIC   -- Z-score (quantos desvios padrão da média)
# MAGIC   ROUND(
# MAGIC     (taxa_alfabetizacao - media_nacional) / NULLIF(desvio_padrao_nacional, 0),
# MAGIC     2
# MAGIC   ) AS z_score,
# MAGIC   
# MAGIC   -- Gap de desigualdade
# MAGIC   ROUND(taxa_maxima_nacional - taxa_minima_nacional, 2) AS gap_desigualdade_nacional
# MAGIC   
# MAGIC FROM ranking
# MAGIC ORDER BY ano DESC, rede, ranking_nacional;
# MAGIC
# MAGIC SELECT '✓ Performance criada: ' || COUNT(*) || ' registros' AS status
# MAGIC FROM workspace.silver.alfabetizacao_performance;

# COMMAND ----------

# DBTITLE 1,3. Indicadores Compostos
# MAGIC %md
# MAGIC ## 3. Indicadores Compostos - Scores e Índices
# MAGIC
# MAGIC Índice consolidado de alfabetização, score de progresso e volatilidade

# COMMAND ----------

# DBTITLE 1,Criar indicadores compostos
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.silver.alfabetizacao_indices_compostos
# MAGIC USING DELTA
# MAGIC AS
# MAGIC WITH evolucao AS (
# MAGIC   SELECT 
# MAGIC     ano,
# MAGIC     sigla_uf,
# MAGIC     rede,
# MAGIC     taxa_alfabetizacao,
# MAGIC     variacao_percentual,
# MAGIC     media_movel_3anos
# MAGIC   FROM workspace.silver.alfabetizacao_evolucao_temporal
# MAGIC ),
# MAGIC performance AS (
# MAGIC   SELECT
# MAGIC     ano,
# MAGIC     sigla_uf,
# MAGIC     rede,
# MAGIC     percentil_performance,
# MAGIC     z_score
# MAGIC   FROM workspace.silver.alfabetizacao_performance
# MAGIC ),
# MAGIC volatilidade AS (
# MAGIC   SELECT 
# MAGIC     sigla_uf,
# MAGIC     rede,
# MAGIC     ano,
# MAGIC     STDDEV(taxa_alfabetizacao) OVER (
# MAGIC       PARTITION BY sigla_uf, rede 
# MAGIC       ORDER BY ano 
# MAGIC       ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
# MAGIC     ) AS volatilidade_3anos
# MAGIC   FROM workspace.bronze.avaliacao_alfabetizacao
# MAGIC   WHERE taxa_alfabetizacao IS NOT NULL
# MAGIC )
# MAGIC SELECT 
# MAGIC   e.ano,
# MAGIC   e.sigla_uf,
# MAGIC   e.rede,
# MAGIC   e.taxa_alfabetizacao,
# MAGIC   
# MAGIC   -- 1. Índice de Alfabetização Consolidado (0-100)
# MAGIC   -- Combina taxa atual + tendência + estabilidade
# MAGIC   ROUND(
# MAGIC     LEAST(100, GREATEST(0,
# MAGIC       (e.taxa_alfabetizacao * 0.7) +                        -- 70% peso na taxa atual
# MAGIC       (COALESCE(e.variacao_percentual, 0) * 2) +           -- 20% peso na evolução (x2)
# MAGIC       (100 - COALESCE(v.volatilidade_3anos, 0) * 10) * 0.1 -- 10% peso na estabilidade
# MAGIC     )),
# MAGIC     2
# MAGIC   ) AS indice_alfabetizacao_consolidado,
# MAGIC   
# MAGIC   -- 2. Score de Progresso (0-100)
# MAGIC   -- Combina posição atual (percentil) + evolução
# MAGIC   ROUND(
# MAGIC     LEAST(100, GREATEST(0,
# MAGIC       p.percentil_performance * 0.6 +                       -- 60% peso no percentil atual
# MAGIC       (50 + COALESCE(e.variacao_percentual, 0) * 2) * 0.4  -- 40% peso na variação (normalizada em torno de 50)
# MAGIC     )),
# MAGIC     2
# MAGIC   ) AS score_progresso,
# MAGIC   
# MAGIC   -- 3. Volatilidade (menor = mais estável)
# MAGIC   ROUND(COALESCE(v.volatilidade_3anos, 0), 2) AS volatilidade_3anos,
# MAGIC   
# MAGIC   -- Classificação de estabilidade
# MAGIC   CASE 
# MAGIC     WHEN COALESCE(v.volatilidade_3anos, 0) < 2 THEN 'MUITO ESTÁVEL'
# MAGIC     WHEN COALESCE(v.volatilidade_3anos, 0) < 5 THEN 'ESTÁVEL'
# MAGIC     WHEN COALESCE(v.volatilidade_3anos, 0) < 10 THEN 'MODERADAMENTE VOLÁTIL'
# MAGIC     ELSE 'ALTA VOLATILIDADE'
# MAGIC   END AS classificacao_estabilidade,
# MAGIC   
# MAGIC   -- 4. Score Final Ponderado (média dos 2 scores)
# MAGIC   ROUND(
# MAGIC     (
# MAGIC       LEAST(100, GREATEST(0, (e.taxa_alfabetizacao * 0.7) + (COALESCE(e.variacao_percentual, 0) * 2) + (100 - COALESCE(v.volatilidade_3anos, 0) * 10) * 0.1)) * 0.5 +
# MAGIC       LEAST(100, GREATEST(0, p.percentil_performance * 0.6 + (50 + COALESCE(e.variacao_percentual, 0) * 2) * 0.4)) * 0.5
# MAGIC     ),
# MAGIC     2
# MAGIC   ) AS score_final_ponderado
# MAGIC   
# MAGIC FROM evolucao e
# MAGIC INNER JOIN performance p ON e.ano = p.ano AND e.sigla_uf = p.sigla_uf AND e.rede = p.rede
# MAGIC LEFT JOIN volatilidade v ON e.ano = v.ano AND e.sigla_uf = v.sigla_uf AND e.rede = v.rede
# MAGIC ORDER BY e.ano DESC, e.rede, score_final_ponderado DESC;
# MAGIC
# MAGIC SELECT '✓ Índices compostos criados: ' || COUNT(*) || ' registros' AS status
# MAGIC FROM workspace.silver.alfabetizacao_indices_compostos;

# COMMAND ----------

# DBTITLE 1,4. Enriquecimento Contextual - Região
# MAGIC %md
# MAGIC ## 4. Enriquecimento Contextual - Dados Regionais
# MAGIC
# MAGIC Join com dados do IBGE (região) e agregação por região

# COMMAND ----------

# DBTITLE 1,Enriquecer com dados regionais
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.silver.alfabetizacao_regional
# MAGIC USING DELTA
# MAGIC AS
# MAGIC WITH base AS (
# MAGIC   SELECT 
# MAGIC     a.ano,
# MAGIC     a.sigla_uf,
# MAGIC     a.rede,
# MAGIC     a.taxa_alfabetizacao,
# MAGIC     u.regiao AS regiao_geografica,
# MAGIC     u.nome AS nome_uf
# MAGIC   FROM workspace.bronze.avaliacao_alfabetizacao a
# MAGIC   INNER JOIN workspace.bronze.uf u ON a.sigla_uf = u.sigla
# MAGIC   WHERE a.taxa_alfabetizacao IS NOT NULL
# MAGIC )
# MAGIC SELECT 
# MAGIC   ano,
# MAGIC   sigla_uf,
# MAGIC   nome_uf,
# MAGIC   regiao_geografica,
# MAGIC   rede,
# MAGIC   taxa_alfabetizacao,
# MAGIC   
# MAGIC   -- Estatísticas regionais
# MAGIC   ROUND(AVG(taxa_alfabetizacao) OVER (PARTITION BY ano, regiao_geografica, rede), 2) AS media_regional,
# MAGIC   ROUND(MIN(taxa_alfabetizacao) OVER (PARTITION BY ano, regiao_geografica, rede), 2) AS min_regional,
# MAGIC   ROUND(MAX(taxa_alfabetizacao) OVER (PARTITION BY ano, regiao_geografica, rede), 2) AS max_regional,
# MAGIC   ROUND(STDDEV(taxa_alfabetizacao) OVER (PARTITION BY ano, regiao_geografica, rede), 2) AS desvio_regional,
# MAGIC   
# MAGIC   -- Posição na região
# MAGIC   RANK() OVER (PARTITION BY ano, regiao_geografica, rede ORDER BY taxa_alfabetizacao DESC) AS ranking_regional,
# MAGIC   
# MAGIC   -- Distância da média regional
# MAGIC   ROUND(
# MAGIC     taxa_alfabetizacao - AVG(taxa_alfabetizacao) OVER (PARTITION BY ano, regiao_geografica, rede),
# MAGIC     2
# MAGIC   ) AS distancia_media_regional,
# MAGIC   
# MAGIC   -- Gap regional (desigualdade interna)
# MAGIC   ROUND(
# MAGIC     MAX(taxa_alfabetizacao) OVER (PARTITION BY ano, regiao_geografica, rede) - 
# MAGIC     MIN(taxa_alfabetizacao) OVER (PARTITION BY ano, regiao_geografica, rede),
# MAGIC     2
# MAGIC   ) AS gap_desigualdade_regional
# MAGIC   
# MAGIC FROM base
# MAGIC ORDER BY ano DESC, regiao_geografica, rede, taxa_alfabetizacao DESC;
# MAGIC
# MAGIC SELECT '✓ Enriquecimento regional criado: ' || COUNT(*) || ' registros' AS status
# MAGIC FROM workspace.silver.alfabetizacao_regional;

# COMMAND ----------

# DBTITLE 1,5. Tabela Consolidada - Feature Store
# MAGIC %md
# MAGIC ## 5. Tabela Consolidada - Feature Store Completo
# MAGIC
# MAGIC Consolida todas as features derivadas em uma única tabela para análise e ML

# COMMAND ----------

# DBTITLE 1,Criar feature store consolidado
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.silver.alfabetizacao_feature_store
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT 
# MAGIC   -- Identificadores
# MAGIC   r.ano,
# MAGIC   r.sigla_uf,
# MAGIC   r.nome_uf,
# MAGIC   r.regiao_geografica,
# MAGIC   r.rede,
# MAGIC   
# MAGIC   -- Métricas Base
# MAGIC   r.taxa_alfabetizacao,
# MAGIC   
# MAGIC   -- Features de Evolução Temporal
# MAGIC   e.variacao_absoluta,
# MAGIC   e.variacao_percentual,
# MAGIC   e.aceleracao,
# MAGIC   e.media_movel_3anos,
# MAGIC   e.desvio_media_movel,
# MAGIC   e.tendencia,
# MAGIC   
# MAGIC   -- Features de Performance
# MAGIC   p.ranking_nacional,
# MAGIC   p.percentil_performance,
# MAGIC   p.quartil,
# MAGIC   p.classificacao_quartil,
# MAGIC   p.media_nacional,
# MAGIC   p.distancia_media_nacional,
# MAGIC   p.z_score,
# MAGIC   p.gap_desigualdade_nacional,
# MAGIC   
# MAGIC   -- Features de Índices Compostos
# MAGIC   i.indice_alfabetizacao_consolidado,
# MAGIC   i.score_progresso,
# MAGIC   i.volatilidade_3anos,
# MAGIC   i.classificacao_estabilidade,
# MAGIC   i.score_final_ponderado,
# MAGIC   
# MAGIC   -- Features Regionais
# MAGIC   r.media_regional,
# MAGIC   r.min_regional,
# MAGIC   r.max_regional,
# MAGIC   r.desvio_regional,
# MAGIC   r.ranking_regional,
# MAGIC   r.distancia_media_regional,
# MAGIC   r.gap_desigualdade_regional
# MAGIC   
# MAGIC FROM workspace.silver.alfabetizacao_regional r
# MAGIC INNER JOIN workspace.silver.alfabetizacao_evolucao_temporal e 
# MAGIC   ON r.ano = e.ano AND r.sigla_uf = e.sigla_uf AND r.rede = e.rede
# MAGIC INNER JOIN workspace.silver.alfabetizacao_performance p 
# MAGIC   ON r.ano = p.ano AND r.sigla_uf = p.sigla_uf AND r.rede = p.rede
# MAGIC INNER JOIN workspace.silver.alfabetizacao_indices_compostos i 
# MAGIC   ON r.ano = i.ano AND r.sigla_uf = i.sigla_uf AND r.rede = i.rede
# MAGIC ORDER BY r.ano DESC, r.rede, i.score_final_ponderado DESC;
# MAGIC
# MAGIC SELECT '✓ Feature Store consolidado criado: ' || COUNT(*) || ' registros, ' || COUNT(DISTINCT sigla_uf) || ' UFs' AS status
# MAGIC FROM workspace.silver.alfabetizacao_feature_store;

# COMMAND ----------

# DBTITLE 1,Validação Final
# MAGIC %md
# MAGIC ## ✅ Validação Final - Tabelas Silver Criadas

# COMMAND ----------

# DBTITLE 1,Validar tabelas silver
# MAGIC %sql
# MAGIC SELECT 
# MAGIC   table_name AS tabela_silver,
# MAGIC   CASE 
# MAGIC     WHEN table_name IN (
# MAGIC       'alfabetizacao_evolucao_temporal',
# MAGIC       'alfabetizacao_performance',
# MAGIC       'alfabetizacao_indices_compostos',
# MAGIC       'alfabetizacao_regional',
# MAGIC       'alfabetizacao_feature_store'
# MAGIC     )
# MAGIC     THEN '✓'
# MAGIC     ELSE ''
# MAGIC   END AS status
# MAGIC FROM system.information_schema.tables
# MAGIC WHERE table_catalog = 'workspace' 
# MAGIC   AND table_schema = 'silver'
# MAGIC   AND table_name LIKE 'alfabetizacao%'
# MAGIC ORDER BY table_name;

# COMMAND ----------

