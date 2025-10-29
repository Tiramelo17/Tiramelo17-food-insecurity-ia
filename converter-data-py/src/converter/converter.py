# python
"""
expand_to_arff.py

Converte um CSV/XLSX com linhas agregadas em um ARFF (.txt).
Uso:
    python expand_to_arff.py                 # usa caminhos hardcoded
    python expand_to_arff.py entrada saida   # sobrescreve caminhos
"""
from typing import List, Optional, Dict, Tuple
import argparse
import pandas as pd
import numpy as np
import math
import sys
import re

# Ajuste estes nomes se necessário
COUNT_COL = "Total_Pessoas_no_Grupo"
INCOME_COL = "Renda_Per_Capita_Estimada"
STATE_COL = "Estado"
STATUS_COL = "Status_Seguranca"

# Colunas de escolaridade no seu CSV (ajuste se nomes diferentes)
EDU_COLS = [
    "Creche",
    "Preescolar",
    "Alfabetizacao_de_jovens_e_adultos",
    "Regular_de_ensino_fundamental",
    "Educacao_de_jovens_e_adultos_do_ensino_fundamental",
    "Regular_do_ensino_medio",
    "Educacao_de_jovens_e_adultos_do_ensino_medio",
    "Superior_de_graduacao",
]

# Mapeamento aproximado de cada categoria de educação para anos de estudo médios
EDU_YEARS_MAP = {
    "Creche": 0.0,
    "Preescolar": 2.0,
    "Alfabetizacao_de_jovens_e_adultos": 4.0,
    "Regular_de_ensino_fundamental": 8.0,
    "Educacao_de_jovens_e_adultos_do_ensino_fundamental": 8.0,
    "Regular_do_ensino_medio": 11.0,
    "Educacao_de_jovens_e_adultos_do_ensino_medio": 11.0,
    "Superior_de_graduacao": 16.0,
}

RACE_COL_CANDIDATES = [
    "Cor",
    "Raca",
    "Cor_Raca",
    "Raca_Cor",
    "Cor_da_Pessoa",
    "Cor_Individuo",
]

# Canonical race values we will emit (plus a fallback 'Desconhecido')
RACE_VALUES = ["Branca", "Preta", "Amarela", "Parda", "Indigena", "Desconhecido"]


def parse_number_br(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating, np.number)):
        try:
            return float(x)
        except Exception:
            return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace("R$", "").replace(" ", "")
    if '.' in s and ',' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '')
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        s = s.replace(',', '.')
    else:
        if s.count('.') > 1:
            s = s.replace('.', '')
    s = re.sub(r"[^0-9.\-]", "", s)
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def build_income_bins(step: int = 500, upto: int = 3000, final_max: int = 50000):
    """
    Constrói bins para o atributo salario.
    Inclui bins especiais sobrepostos para garantir que existam os ranges:
      422-999, 810-1300, 1500-3000
    e depois os bins regulares de step=500.
    Retorna lista de (low, high, label) — ordem importante: bins especiais primeiro.
    """
    special = [
        (422, 999, "422-999"),
        (810, 1300, "810-1300"),
        (1500, 3000, "1500-3000"),
    ]
    bins = []
    # add special bins first (avoid duplicates)
    seen_labels = set()
    for low, high, label in special:
        bins.append((low, high, label))
        seen_labels.add(label)

    low = 0
    while low < upto:
        high = low + step - 1
        label = f"{low}-{high}"
        if label not in seen_labels:
            bins.append((low, high, label))
            seen_labels.add(label)
        low += step

    # final large bin
    final_label = f"{upto}-{final_max}"
    if final_label not in seen_labels:
        bins.append((upto, final_max, final_label))
    return bins


def income_to_bin_label(value, bins):
    """
    Mapeia valor numérico para o primeiro bin que contenha o valor.
    Valores faltantes -> "?".
    """
    try:
        if value is None:
            return "?"
        if isinstance(value, str):
            v = parse_number_br(value)
        else:
            v = float(value)
    except Exception:
        return "?"
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "?"
    if v < 0:
        v = 0.0
    # procura o primeiro bin que contenha v (por isso bins especiais vêm primeiro)
    for low, high, label in bins:
        if v >= low and v <= high:
            return label
    return bins[-1][2]


def sanitize_nominal_value(v: str) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    s = s.replace("{", "").replace("}", "").replace(",", "").replace("'", "")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-_]", "", s, flags=re.UNICODE)
    return s if s != "" else "Desconhecido"


def sanitize_race_value(v: str) -> str:
    if v is None:
        return "Desconhecido"
    s = str(v).strip().lower()
    if s == "" or s in ["nao informado", "n/a", "na", "null"]:
        return "Desconhecido"
    import unicodedata
    s_noacc = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    if any(x in s_noacc for x in ["branc", "branco"]):
        return "Branca"
    if any(x in s_noacc for x in ["pret", "preta", "negro", "negra"]):
        return "Preta"
    if any(x in s_noacc for x in ["amarel", "amarela"]):
        return "Amarela"
    if any(x in s_noacc for x in ["pard", "parda", "pardo"]):
        return "Parda"
    if any(x in s_noacc for x in ["indigen", "indigena", "indig"]):
        return "Indigena"
    for cand in ["branca", "preta", "amarela", "parda", "indigena"]:
        if cand in s_noacc:
            return cand.capitalize()
    return "Desconhecido"


def compute_expected_years_for_row(row: pd.Series, edu_cols: List[str]) -> float:
    counts = []
    for c in edu_cols:
        val = row.get(c, 0)
        try:
            if pd.isna(val) or val == "":
                ival = 0
            else:
                ival = int(str(val).replace(".", "").replace(",", ""))
        except Exception:
            try:
                ival = int(float(val))
            except Exception:
                ival = 0
        counts.append(ival)
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    years = np.array([EDU_YEARS_MAP.get(c, 0.0) for c in edu_cols], dtype=float)
    expected = float((counts * years).sum() / total)
    return expected


def parse_int_count(x) -> int:
    try:
        if pd.isna(x):
            return 0
        s = str(x).strip()
        if s == "":
            return 0
        s2 = s.replace(".", "").replace(",", "")
        return int(s2)
    except Exception:
        try:
            return int(float(str(x)))
        except Exception:
            return 0


def expand_row(row: pd.Series,
               edu_cols: List[str],
               count_col: str,
               income_col: str,
               state_col: str,
               status_col: str,
               edu_sample: bool,
               rng: np.random.RandomState,
               income_noise_std: float = 0.0,
               race_col: Optional[str] = None,
               race_count_cols: Optional[List[str]] = None,
               race_labels: Optional[List[str]] = None):
    try:
        n_people = int(str(row[count_col]).replace(".", "").replace(",", ""))
    except:
        n_people = 0
    if n_people <= 0:
        return []

    income = parse_number_br(row.get(income_col, np.nan))
    if income_noise_std and (not math.isnan(income)) and income_noise_std > 0:
        incomes = rng.normal(loc=income, scale=income_noise_std, size=n_people)
        incomes = np.clip(incomes, a_min=0.0, a_max=None)
    else:
        incomes = np.full(n_people, income if not pd.isna(income) else float("nan"))

    state_raw = row.get(state_col, "Desconhecido")
    state = sanitize_nominal_value(state_raw)

    race = "Desconhecido"
    per_person_race = None
    if race_count_cols is not None and len(race_count_cols) > 0:
        rcnts = []
        for col in race_count_cols:
            v = row.get(col, 0)
            if pd.isna(v) or v == "":
                iv = 0
            else:
                sval = str(v).replace(".", "").replace(",", "")
                try:
                    iv = int(sval)
                except Exception:
                    try:
                        iv = int(float(sval))
                    except Exception:
                        iv = 0
            rcnts.append(max(0, iv))
        rcnts = np.array(rcnts, dtype=float)
        total_rc = rcnts.sum()
        if total_rc > 0:
            probs = rcnts / total_rc
            chosen = rng.choice(len(rcnts), size=n_people, p=probs)
            unknown_race = sanitize_race_value(None)
            per_person_race = [race_labels[i] if race_labels and i < len(race_labels) else unknown_race for i in np.atleast_1d(chosen)]
        else:
            per_person_race = ["Desconhecido"] * n_people
    else:
        if race_col is not None:
            race_raw = row.get(race_col, None)
            race = sanitize_race_value(race_raw)

    status_raw = str(row.get(status_col, "")).strip().lower()
    inseg = "sim" if "inseg" in status_raw else "nao"

    edu_counts = []
    for c in edu_cols:
        val = row.get(c, 0)
        if pd.isna(val) or val == "":
            ival = 0
        else:
            sval = str(val).replace(".", "").replace(",", "")
            try:
                ival = int(sval)
            except Exception:
                try:
                    ival = int(float(sval))
                except Exception:
                    ival = 0
        edu_counts.append(max(0, ival))
    edu_counts = np.array(edu_counts, dtype=int)
    total_edu = edu_counts.sum()

    if edu_sample and total_edu > 0:
        p = edu_counts / total_edu
        chosen = rng.choice(len(edu_cols), size=n_people, p=p)
        edu_years = np.array([EDU_YEARS_MAP.get(edu_cols[i], float("nan")) for i in chosen], dtype=float)
    else:
        expected_years = compute_expected_years_for_row(row, edu_cols)
        edu_years = np.full(n_people, expected_years, dtype=float)

    result = []
    for i in range(n_people):
        sal = incomes[i]
        ey = edu_years[i]
        race_i = per_person_race[i] if per_person_race is not None else race
        result.append((sal, state, race_i, ey, inseg))
    return result


def write_arff_header(f, relation_name: str, states: List[str], races: List[str]):
    f.write(f"@relation {relation_name}\n\n")
    income_bins = build_income_bins(step=500, upto=3000, final_max=50000)
    income_labels = ",".join([b[2] for b in income_bins])
    f.write(f"@attribute salario {{{income_labels}}}\n")
    states_list = ",".join(states)
    f.write(f"@attribute estado {{{states_list}}}\n")
    races_list = ",".join(races)
    f.write(f"@attribute cor {{{races_list}}}\n")
    f.write("@attribute escolaridade numeric\n")
    f.write("@attribute inseguranca {sim,nao}\n\n")
    f.write("@data\n")


def main():
    DEFAULT_INPUT = "/home/gabriel/Downloads/base_granular_para_ml.xlsx"
    DEFAULT_OUTPUT = "/home/gabriel/Downloads/base_granular_para_ml.arff"
    DEFAULT_MAX_ROWS = 2_000_000
    DEFAULT_MAX_PER_STATE = 1_000_000

    parser = argparse.ArgumentParser(description="Expand aggregated CSV/XLSX into ARFF for Weka.")
    parser.add_argument("input_csv", nargs="?", help="CSV/XLSX de entrada (com cabeçalho).", default=DEFAULT_INPUT)
    parser.add_argument("output_arff", nargs="?", help="Arquivo de saída (.arff ou .txt).", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-edu-sample", dest="edu_sample", action="store_false",
                        help="Não amostrar escolaridade por pessoa; usar média esperada por linha.")
    parser.add_argument("--edu-sample", dest="edu_sample", action="store_true",
                        help="Amostrar escolaridade por pessoa (prob. proporcional às contagens).")
    parser.set_defaults(edu_sample=False)
    parser.add_argument("--sample-fraction", type=float, default=None,
                        help="Apenas amostrar essa fração (0..1) do grupo antes de expandir.")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                        help="Máximo total de linhas a escrever (após expansão). Default é 2_000_000.")
    parser.add_argument("--max-per-state", type=int, default=DEFAULT_MAX_PER_STATE,
                        help="Máximo de linhas por estado (após expansão). Default é 1_000_000.")
    parser.add_argument("--income-noise-std", type=float, default=0.0,
                        help="Adicionar ruído na renda (desvio padrão).")
    parser.add_argument("--seed", type=int, default=42, help="Seed aleatória.")
    args = parser.parse_args()

    if args.input_csv == DEFAULT_INPUT and args.output_arff == DEFAULT_OUTPUT:
        print(f"Usando caminhos padrão: entrada=`{DEFAULT_INPUT}` saída=`{DEFAULT_OUTPUT}`")
    if args.max_rows == DEFAULT_MAX_ROWS:
        print(f"Usando limite máximo de linhas: {DEFAULT_MAX_ROWS}")
    if args.max_per_state == DEFAULT_MAX_PER_STATE:
        print(f"Usando limite por estado: {DEFAULT_MAX_PER_STATE}")

    try:
        input_path = args.input_csv
        if input_path.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(input_path, sheet_name=0)
            df = df.fillna("").astype(str)
        else:
            df = pd.read_csv(input_path, dtype=str).fillna("")
    except Exception as e:
        print("Erro ao ler arquivo:", e)
        sys.exit(1)

    if COUNT_COL not in df.columns:
        print(f"Coluna esperada '{COUNT_COL}' não encontrada no arquivo.")
        sys.exit(1)

    # Preprocessa: contagem numérica, estado sanitizado, status binário
    df['_count'] = df[COUNT_COL].apply(parse_int_count)
    df['_state'] = df.get(STATE_COL, pd.Series(["Desconhecido"] * len(df))).apply(sanitize_nominal_value)
    df['_status'] = df.get(STATUS_COL, pd.Series([""] * len(df))).apply(lambda s: "sim" if "inseg" in str(s).lower() else "nao")

    # Monta lista de estados únicos para header (sanitizados)
    states_unique = []
    seen = set()
    for s in df['_state'].tolist():
        if s not in seen:
            seen.add(s)
            states_unique.append(s)
    if len(states_unique) == 0:
        states_unique = ["Desconhecido"]

    # Detecta informações de cor/raça (igual antes)
    race_col = None
    race_count_cols = []
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in RACE_COL_CANDIDATES:
        if cand.lower() in cols_lower:
            race_col = cols_lower[cand.lower()]
            break
    race_labels_present = []
    for label in ["Branca", "Preta", "Amarela", "Parda", "Indigena"]:
        if label.lower() in cols_lower:
            race_count_cols.append(cols_lower[label.lower()])
            race_labels_present.append(label)

    # build race values for header
    if len(race_count_cols) > 0:
        races_ordered = [r for r in ["Branca", "Preta", "Amarela", "Parda", "Indigena"] if r in race_labels_present]
        for r in RACE_VALUES:
            if r not in races_ordered:
                races_ordered.append(r)
    else:
        races_found = []
        if race_col is not None:
            for raw in df[race_col].tolist():
                r = sanitize_race_value(raw)
                if r not in races_found:
                    races_found.append(r)
        races_ordered = [r for r in RACE_VALUES if r in races_found]
        for r in RACE_VALUES:
            if r not in races_ordered:
                races_ordered.append(r)

    # Calcula totais por estado+status
    grouped = df.groupby(['_state', '_status'])['_count'].sum().reset_index()
    # Inicializa alocação por estado-status
    alloc: Dict[Tuple[str, str], int] = {}
    # Para cada estado, decide quota por status
    for state, g in grouped.groupby('_state'):
        state_totals = g.set_index('_status')['_count'].to_dict()
        statuses = list(state_totals.keys())
        if 'sim' in statuses and 'nao' in statuses:
            half = args.max_per_state // 2
            # se ímpar, dá 1 extra para 'sim'
            alloc[(state, 'sim')] = min(state_totals.get('sim', 0), half + (args.max_per_state % 2))
            alloc[(state, 'nao')] = min(state_totals.get('nao', 0), half)
        else:
            # apenas um status presente -> recebe todo o limite por estado (ou o total, se menor)
            only_status = statuses[0]
            alloc[(state, only_status)] = min(state_totals.get(only_status, 0), args.max_per_state)

    # se faltar alguma combinação no alloc, inicializa com 0
    for st in states_unique:
        for stt in ['sim', 'nao']:
            alloc.setdefault((st, stt), 0)

    rng = np.random.RandomState(args.seed)
    relation_name = "inseguranca_alimentar"
    with open(args.output_arff, "w", encoding="utf-8") as fout:
        write_arff_header(fout, relation_name, states_unique, races_ordered)
        income_bins = build_income_bins(step=500, upto=3000, final_max=50000)

        total_written = 0
        # percorre as linhas originais; para cada uma, corta a contagem conforme alocação do seu estado+status
        for idx, row in df.iterrows():
            orig_n = int(row['_count'])
            if orig_n <= 0:
                continue

            state = row['_state']
            status = row['_status']

            # aplica sample_fraction antes de contar
            if args.sample_fraction is not None and 0.0 < args.sample_fraction < 1.0:
                n_people = int(rng.binomial(orig_n, args.sample_fraction))
            else:
                n_people = orig_n
            if n_people <= 0:
                continue

            # limita por alocação do estado+status
            remaining_alloc = alloc.get((state, status), 0)
            if remaining_alloc <= 0:
                continue
            take = min(n_people, remaining_alloc)

            # também respeita o limite global restante
            if args.max_rows is not None:
                global_remaining = args.max_rows - total_written
                if global_remaining <= 0:
                    break
                take = min(take, global_remaining)

            if take <= 0:
                continue

            # cria cópia local da linha com contagem ajustada
            row_loc = row.copy()
            row_loc[COUNT_COL] = str(take)
            # expandir
            expanded = expand_row(row_loc, EDU_COLS, COUNT_COL, INCOME_COL, STATE_COL, STATUS_COL,
                                  edu_sample=args.edu_sample, rng=rng, income_noise_std=args.income_noise_std,
                                  race_col=race_col, race_count_cols=race_count_cols, race_labels=race_labels_present)

            for sal, state_s, race, ey, inseg in expanded:
                # manipulações aleatórias por status:
                rdraw = rng.random()
                if inseg == "sim":
                    # 3% -> 810-1300, next 15% -> 422-999 (total 18% com ordem definida)
                    if rdraw < 0.03:
                        sal = rng.uniform(810, 1300)
                    elif rdraw < 0.18:
                        sal = rng.uniform(422, 999)
                else:
                    # para 'nao' 15% -> 1500-3000
                    if rdraw < 0.15:
                        sal = rng.uniform(1500, 3000)

                sal_str = income_to_bin_label(sal, income_bins)
                ey_str = "?" if (ey is None or (isinstance(ey, float) and math.isnan(ey))) else ("{:.2f}".format(float(ey)))
                state_str = state_s if state_s in states_unique else states_unique[0]
                race_str = race if race in races_ordered else races_ordered[-1]
                fout.write(f"{sal_str},{state_str},{race_str},{ey_str},{inseg}\n")
                total_written += 1
                # decrementa alocação por estado+status
                alloc[(state, status)] = max(0, alloc.get((state, status), 0) - 1)
                if args.max_rows is not None and total_written >= args.max_rows:
                    break
            if args.max_rows is not None and total_written >= args.max_rows:
                break

    print(f"ARFF gerado: {args.output_arff} (linhas escritas: {total_written})")


if __name__ == "__main__":
    main()
