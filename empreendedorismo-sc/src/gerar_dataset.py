"""
Gerador de Dataset Sintético - Empreendedorismo em Santa Catarina
=================================================================
Fonte de referência: SEBRAE-SC, JUCESC, IBGE Cidades SC, RAIS/MTE
Os dados são sintéticos, porém calibrados com base em estatísticas
reais publicadas por essas fontes (proporções, médias, distribuições).

Referências utilizadas para calibração:
- SEBRAE-SC: Relatório de Sobrevivência de Empresas (2023)
- JUCESC: Boletim de Abertura de Empresas SC (2022-2023)
- IBGE: Cadastro Central de Empresas (CEMPRE) SC 2021
- RAIS/MTE: Vínculos formais por setor em SC (2022)
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

# ── Parâmetros calibrados com dados públicos ──────────────────────────────────

MUNICIPIOS = {
    "Florianópolis":  {"pop": 508_000, "peso": 0.14},
    "Joinville":      {"pop": 616_000, "peso": 0.16},
    "Blumenau":       {"pop": 360_000, "peso": 0.10},
    "São José":       {"pop": 250_000, "peso": 0.07},
    "Chapecó":        {"pop": 230_000, "peso": 0.07},
    "Itajaí":         {"pop": 230_000, "peso": 0.07},
    "Criciúma":       {"pop": 215_000, "peso": 0.06},
    "Jaraguá do Sul": {"pop": 180_000, "peso": 0.05},
    "Palhoça":        {"pop": 175_000, "peso": 0.05},
    "Balneário Camboriú":{"pop": 140_000,"peso": 0.04},
    "Lages":          {"pop": 156_000, "peso": 0.04},
    "Brusque":        {"pop": 140_000, "peso": 0.04},
    "Tubarão":        {"pop": 110_000, "peso": 0.03},
    "São Bento do Sul":{"pop": 85_000, "peso": 0.03},
    "Camboriú":       {"pop": 75_000,  "peso": 0.05},
}

SETORES = {
    "Comércio Varejista":        {"cnae": "47", "peso": 0.28, "media_func": 4.2,  "media_fat": 280_000},
    "Serviços":                  {"cnae": "74", "peso": 0.22, "media_func": 3.1,  "media_fat": 185_000},
    "Alimentação e Hospedagem":  {"cnae": "56", "peso": 0.12, "media_func": 5.8,  "media_fat": 320_000},
    "Construção Civil":          {"cnae": "41", "peso": 0.10, "media_func": 6.5,  "media_fat": 410_000},
    "Indústria Têxtil":          {"cnae": "13", "peso": 0.07, "media_func": 12.4, "media_fat": 680_000},
    "Tecnologia da Informação":  {"cnae": "62", "peso": 0.07, "media_func": 4.8,  "media_fat": 520_000},
    "Agronegócio":               {"cnae": "01", "peso": 0.05, "media_func": 3.0,  "media_fat": 290_000},
    "Saúde e Bem-estar":         {"cnae": "86", "peso": 0.05, "media_func": 3.5,  "media_fat": 240_000},
    "Indústria Metalmecânica":   {"cnae": "25", "peso": 0.04, "media_func": 18.0, "media_fat": 920_000},
}

PORTES = {
    "MEI":    {"peso": 0.52, "faixa_fat": (0,       81_000),  "max_func": 1},
    "ME":     {"peso": 0.31, "faixa_fat": (81_001,  360_000), "max_func": 19},
    "EPP":    {"peso": 0.12, "faixa_fat": (360_001, 4_800_000),"max_func": 99},
    "Médio":  {"peso": 0.03, "faixa_fat": (4_800_001,300_000_000),"max_func": 499},
    "Grande": {"peso": 0.02, "faixa_fat": (300_000_001, 1e9), "max_func": 9999},
}

FORMAS_JURIDICAS = ["MEI", "LTDA", "S/A", "EI", "SLU", "Associação/Coop."]
ANOS_ABERTURA = list(range(2010, 2024))

N = 3_000


def gerar_dataset(n: int = N) -> pd.DataFrame:
    municipios_lista = list(MUNICIPIOS.keys())
    pesos_mun = [MUNICIPIOS[m]["peso"] for m in municipios_lista]

    setores_lista = list(SETORES.keys())
    pesos_set = [SETORES[s]["peso"] for s in setores_lista]

    portes_lista = list(PORTES.keys())
    pesos_porte = [PORTES[p]["peso"] for p in portes_lista]

    registros = []
    for i in range(n):
        municipio = np.random.choice(municipios_lista, p=pesos_mun)
        setor     = np.random.choice(setores_lista, p=pesos_set)
        porte     = np.random.choice(portes_lista, p=pesos_porte)
        ano_abertura = np.random.choice(ANOS_ABERTURA)

        setor_info = SETORES[setor]
        porte_info = PORTES[porte]

        # Faturamento anual (R$)
        fat_min, fat_max = porte_info["faixa_fat"]
        faturamento = np.random.lognormal(
            mean=np.log(max(setor_info["media_fat"], fat_min + 1)),
            sigma=0.6
        )
        faturamento = float(np.clip(faturamento, fat_min, fat_max))

        # Funcionários
        max_f = min(porte_info["max_func"], setor_info["media_func"] * 4)
        funcionarios = int(np.clip(
            np.random.poisson(setor_info["media_func"]), 0, max_f
        ))
        if porte == "MEI":
            funcionarios = min(funcionarios, 1)

        # Idade da empresa em anos
        idade = 2024 - ano_abertura

        # Acesso a crédito
        prob_credito = 0.25 + (0.15 if porte in ["EPP","Médio","Grande"] else 0)
        acesso_credito = np.random.binomial(1, prob_credito)

        # Participação em programa de apoio (SEBRAE, BADESC, SC Inova etc.)
        prob_prog = 0.20 + (0.10 if setor == "Tecnologia da Informação" else 0)
        participou_programa = np.random.binomial(1, prob_prog)

        # Exporta?
        prob_exp = 0.02
        if setor in ["Indústria Têxtil","Indústria Metalmecânica","Agronegócio","Tecnologia da Informação"]:
            prob_exp = 0.12
        if porte in ["Médio","Grande"]:
            prob_exp += 0.15
        exporta = np.random.binomial(1, prob_exp)

        # Nível de inovação (0-10)
        base_inov = {"Tecnologia da Informação": 7, "Indústria Metalmecânica": 5,
                     "Saúde e Bem-estar": 5, "Agronegócio": 4}.get(setor, 3)
        inovacao = int(np.clip(np.random.normal(base_inov, 1.5), 0, 10))

        # Forma jurídica coerente com porte
        if porte == "MEI":
            forma_jur = "MEI"
        elif porte == "ME":
            forma_jur = np.random.choice(["LTDA","EI","SLU"], p=[0.55,0.30,0.15])
        elif porte == "EPP":
            forma_jur = np.random.choice(["LTDA","SLU","S/A"], p=[0.70,0.20,0.10])
        else:
            forma_jur = np.random.choice(["LTDA","S/A"], p=[0.60,0.40])

        # Situação cadastral (ativa / baixada / suspensa)
        # Taxa de sobrevivência de 5 anos no Brasil ≈ 60% (SEBRAE 2023)
        prob_ativa = 0.88 if idade <= 2 else (0.72 if idade <= 5 else 0.62)
        situacao = np.random.choice(
            ["Ativa","Baixada","Suspensa"],
            p=[prob_ativa, (1-prob_ativa)*0.80, (1-prob_ativa)*0.20]
        )

        # Região do estado
        regiao_map = {
            "Florianópolis": "Grande Florianópolis",
            "São José": "Grande Florianópolis",
            "Palhoça": "Grande Florianópolis",
            "Joinville": "Norte",
            "Jaraguá do Sul": "Norte",
            "São Bento do Sul": "Norte",
            "Blumenau": "Vale do Itajaí",
            "Brusque": "Vale do Itajaí",
            "Itajaí": "Vale do Itajaí",
            "Balneário Camboriú": "Vale do Itajaí",
            "Camboriú": "Vale do Itajaí",
            "Chapecó": "Oeste",
            "Criciúma": "Sul",
            "Tubarão": "Sul",
            "Lages": "Serra",
        }
        regiao = regiao_map.get(municipio, "Outras")

        # Introduzir valores ausentes (realismo)
        if np.random.rand() < 0.04:
            faturamento = np.nan
        if np.random.rand() < 0.03:
            funcionarios = np.nan
        if np.random.rand() < 0.02:
            inovacao = np.nan

        # Variável-alvo: SOBREVIVÊNCIA (empresa ativa após 3 anos)
        # Baseada em logit com fatores reais
        score = (
            0.8 * acesso_credito +
            0.6 * participou_programa +
            0.5 * exporta +
            0.04 * inovacao if not np.isnan(inovacao) else 0 +
            0.02 * (funcionarios if not np.isnan(funcionarios) else 0) +
            (-0.5 if situacao == "Baixada" else 0) +
            np.random.normal(0, 0.3)
        )
        sobreviveu = 1 if (situacao == "Ativa" and score > -0.2) else 0

        registros.append({
            "id_empresa":          f"SC{i+1:05d}",
            "municipio":           municipio,
            "regiao":              regiao,
            "setor":               setor,
            "cnae_grupo":          setor_info["cnae"],
            "porte":               porte,
            "forma_juridica":      forma_jur,
            "ano_abertura":        ano_abertura,
            "idade_anos":          idade,
            "faturamento_anual":   round(faturamento, 2) if not np.isnan(faturamento) else np.nan,
            "num_funcionarios":    funcionarios,
            "acesso_credito":      acesso_credito,
            "participou_programa": participou_programa,
            "exporta":             exporta,
            "nivel_inovacao":      inovacao,
            "situacao_cadastral":  situacao,
            "sobreviveu_3anos":    sobreviveu,
        })

    return pd.DataFrame(registros)


if __name__ == "__main__":
    df = gerar_dataset()
    df.to_csv("SUA PASTA/empreendedorismo-sc/dados/empresas_sc.csv", index=False)
    print(f"Dataset gerado: {len(df)} registros, {df.columns.tolist()}")
    print(df.head())
