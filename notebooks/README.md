# Notebooks

02 — Silver: Engenharia de Atributos (Feature Engineering)
Camada Silver: transforma dados brutos do Bronze em features analíticas derivadas.

Atributos Derivados:
📊 Evolução Temporal
Variação ano a ano (delta e percentual)
Média móvel (últimos 3 anos)
Tendência linear
Aceleração (variação da variação)
🎯 Indicadores de Performance
Distância para meta nacional/estadual
Ranking entre UFs
Classificação por quartis
Gap de desigualdade (diferença entre maior e menor por ano)
🔄 Indicadores Compostos
Índice de Alfabetização Consolidado (média ponderada por nível)
Score de Progresso (evolução + posição atual)
Volatilidade (desvio padrão últimos 3 anos)
🌐 Enriquecimento Contextual
Join com dados PNAD (contexto socioeconômico)
Join com dados FUNDEB (investimento em educação)
Densidade populacional (UF)
Região geográfica

1. Evolução Temporal - Métricas por UF e Ano
Calcular variações ano a ano, tendências e médias móveis

2. Indicadores de Performance - Rankings e Metas
Ranking entre UFs, distância para meta, classificação por quartis

3. Indicadores Compostos - Scores e Índices
Índice consolidado de alfabetização, score de progresso e volatilidade

4. Enriquecimento Contextual - Dados Regionais
Join com dados do IBGE (região) e agregação por região

5. Tabela Consolidada - Feature Store Completo
Consolida todas as features derivadas em uma única tabela para análise e ML

 Validação Final - Tabelas Silver Criadas