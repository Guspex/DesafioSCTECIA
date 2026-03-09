# 📊 Empreendedorismo em Santa Catarina — Análise e Modelo Preditivo

## Descrição do Problema

Santa Catarina é um dos estados com maior dinamismo econômico do Brasil, concentrando um
ecossistema diversificado de micro e pequenos empreendimentos, startups de tecnologia,
indústrias têxteis e metalmecânicas, além de negócios no agronegócio e no turismo.

Este projeto tem como objetivo **organizar, analisar e construir um modelo preditivo** sobre
o cenário empreendedor catarinense, respondendo à pergunta central:

> **Quais fatores influenciam a sobrevivência de uma empresa em Santa Catarina após 3 anos
> de operação, e é possível prever esse resultado com base em dados cadastrais e
> socioeconômicos?**

A variável-alvo escolhida é binária: `sobreviveu_3anos` (1 = empresa ativa após 3 anos;
0 = encerrada ou suspensa), permitindo o treinamento de um **modelo de classificação
supervisionada**.

---

## Origem e Natureza dos Dados

> ⚠️ **Importante:** O conjunto de dados utilizado é **sintético**, porém calibrado com
> base em estatísticas publicadas por fontes públicas oficiais, conforme detalhado abaixo.

### Fontes de referência utilizadas para calibração

|                               Fonte                                      |                          Informação utilizada                          |        Acesso        |
|--------------------------------------------------------------------------|------------------------------------------------------------------------|----------------------|
| **SEBRAE-SC** — Relatório de Sobrevivência de Empresas (2023)            | Taxa de sobrevivência de 5 anos (~60%), distribuição por porte e setor | sebrae.com.br/sc     |
| **JUCESC** — Boletim de Abertura/Encerramento de Empresas SC (2022-2023) | Volume de abertura por município, distribuição de formas jurídicas     | jucesc.com.br        |
| **IBGE — CEMPRE** (Cadastro Central de Empresas SC, 2021)                | Estrutura de porte, faixas de faturamento, número de funcionários      | ibge.gov.br/cempre   |
| **RAIS/MTE** — Vínculos empregatícios formais por setor em SC (2022)     | Médias de funcionários por setor, distribuição regional                | gov.br/trabalho/rais |

Os dados sintéticos respeitam as seguintes proporções reais documentadas:
- ~52–54% das empresas catarinenses são MEIs (SEBRAE-SC, 2023)
- Taxa de sobrevivência de 3 anos: 65–70% (SEBRAE Nacional, 2023)
- Distribuição regional concentrada no Norte (Joinville) e Vale do Itajaí (Blumenau)
- Média de 4–5 funcionários em MPEs do setor de Serviços

---

## Etapas Realizadas

### 1. Geração do Dataset (`src/gerar_dataset.py`)

O script constrói um dataset com **3.000 registros** de empresas fictícias,
parametrizados pelas distribuições reais supracitadas. Cada registro contém:

- Identificação: `id_empresa`, `municipio`, `regiao`, `setor`, `cnae_grupo`
- Características: `porte`, `forma_juridica`, `ano_abertura`, `idade_anos`
- Financeiro: `faturamento_anual`, `num_funcionarios`
- Fatores de apoio: `acesso_credito`, `participou_programa`, `exporta`
- Qualitativo: `nivel_inovacao` (escala 0–10)
- Situação: `situacao_cadastral`, **`sobreviveu_3anos`** (variável-alvo)

Valores ausentes foram inseridos artificialmente em ~3–4% dos registros numéricos para
simular imperfeições reais de bases governamentais.

### 2. Carregamento e Diagnóstico

- Leitura do CSV com `pandas`
- Verificação de tipos, valores ausentes e duplicatas
- Análise de outliers via método IQR (3×IQR)

### 3. Tratamento de Dados

- **Imputação** de valores ausentes pela mediana do setor (estratégia robusta a outliers)
- **Engenharia de variáveis:**
  - `log_faturamento`: transformação logarítmica para reduzir assimetria
  - `empresa_madura`: flag binária para empresas com ≥ 5 anos
  - `faixa_etaria`: segmentação etária da empresa (0–2, 3–5, 6–10, 11+ anos)

### 4. Análise Exploratória (EDA)

Foram gerados **8 gráficos** salvos na pasta `resultados/`:

|              Arquivo              |                  Conteúdo                       |
|-----------------------------------|-------------------------------------------------|
| `01_distribuicao_porte_setor.png` | Distribuição por porte e setor                  |
| `02_regioes_sobrevivencia.png`    | Volume e taxa de sobrevivência por região       |
| `03_indicadores_setor.png`        | Faturamento mediano e funcionários por setor    |
| `04_correlacao.png`               | Matriz de correlação entre variáveis numéricas  |
| `05_fatores_sobrevivencia.png`    | Impacto de crédito e programas de apoio         |
| `06_roc_confusao.png`             | Curva ROC e Matriz de Confusão do melhor modelo |
| `07_importancia_variaveis.png`    | Importância das variáveis (Random Forest)       |
| `08_comparacao_modelos.png`       | AUC comparativo entre os 3 modelos              |

### 5. Modelagem Preditiva

**Definição da variável-alvo:** `sobreviveu_3anos` (classificação binária)

**Pipeline de pré-processamento:**
- Variáveis numéricas: imputação pela mediana → normalização (StandardScaler)
- Variáveis categóricas: imputação pela moda → One-Hot Encoding

**Modelos avaliados:**
1. Regressão Logística (baseline interpretável)
2. Random Forest (ensemble, robusto a não-linearidades)
3. Gradient Boosting (boosting sequencial)

**Validação:** Stratified K-Fold (5 splits), métrica principal: ROC-AUC

### 6. Resultados e Interpretação

|        Modelo       | AUC CV (média) | AUC Teste  |
|---------------------|----------------|------------|
| Regressão Logística | 0.5936         | 0.5671     |
| **Random Forest**   | **0.6032**     | **0.5829** |
| Gradient Boosting   | 0.5956         | 0.5678     |

**Melhor modelo:** Random Forest — AUC: 0.583 | F1: 0.594 | Acurácia: 0.547

**Interpretação do desempenho:** O AUC moderado (~0.58) reflete um cenário realista —
a sobrevivência empresarial é determinada por fatores não capturáveis exclusivamente por
dados cadastrais (contexto macroeconômico, habilidades do empreendedor, decisões táticas).
Ainda assim, o modelo supera a predição aleatória (AUC = 0.50) e identifica padrões úteis.

**Principais insights:**
- Empresas com **acesso a crédito** têm taxa de sobrevivência ~1.8 p.p. maior
- **Participação em programas** (SEBRAE, SC Inova, BADESC) eleva a taxa de sobrevivência
- O setor de **Construção Civil** apresenta a menor taxa de sobrevivência (61.3%)
- **Grandes empresas** apresentam maior taxa (78.6%), evidenciando vantagem de escala
- Região **Grande Florianópolis** concentra maior densidade de empresas de TI e serviços

---

## Tecnologias Utilizadas

| Biblioteca     | Versão |             Finalidade                 |
|----------------|--------|----------------------------------------|
| `Python`       | 3.12   | Linguagem principal                    |
| `pandas`       | 2.x    | Manipulação e análise de dados         |
| `numpy`        | 1.x    | Operações numéricas e geração de dados |
| `scikit-learn` | 1.x    | Pipeline de ML, modelos e métricas     |
| `matplotlib`   | 3.x    | Geração de gráficos                    |
| `seaborn`      | 0.13.x | Visualizações estatísticas             |

---

## Estrutura do Projeto

```
DesafioSCTECIA/
|
├── LICENSE
├── README.md 
| 
└── empreendedorismo-sc/
    ├── requirements.txt                   # Dependências do projeto
    │
    ├── dados/
    │   └── empresas_sc.csv                # Dataset gerado (3.000 registros)
    │
    ├── src/
    │   ├── gerar_dataset.py               # Gerador do dataset sintético calibrado
    │   └── analise_principal.py           # Pipeline completo de análise e modelagem
    │
    └── resultados/
        ├── 01_distribuicao_porte_setor.png
        ├── 02_regioes_sobrevivencia.png
        ├── 03_indicadores_setor.png
        ├── 04_correlacao.png
        ├── 05_fatores_sobrevivencia.png
        ├── 06_roc_confusao.png
        ├── 07_importancia_variaveis.png
        ├── 08_comparacao_modelos.png
        ├── tab_resumo_porte.csv
        ├── tab_resumo_setor.csv
        └── tab_metricas_modelos.csv

```

---

## Como Executar

### Pré-requisitos

- Python 3.10+ instalado
- pip disponível

### Passo a passo

```bash
# 1. Clone ou extraia o projeto
cd empreendedorismo-sc

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Gere o dataset (caso não exista)
python src/gerar_dataset.py

# 4. Execute a análise completa
python src/analise_principal.py
```

Os gráficos serão salvos em `resultados/` e os resumos tabulares em CSV.

---

## Considerações Finais

Esta solução demonstra um fluxo completo de **ciência de dados aplicada ao
empreendedorismo catarinense**, desde a geração e tratamento dos dados até a
interpretação de um modelo preditivo. Embora o dataset seja sintético, ele foi
construído com rigor estatístico a partir de referências públicas, tornando os
resultados coerentes com a realidade observada no estado.

Trabalhos futuros podem incorporar dados reais da JUCESC via API pública,
cruzar com indicadores do IBGE por município ou integrar dados de crédito do
Banco do Brasil/BADESC para elevar o poder preditivo do modelo.

---

*Projeto desenvolvido como exercício prático de ciência de dados — SC, Março de 2026.*
