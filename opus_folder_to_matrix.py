#!/usr/bin/env python3
"""
opus_folder_to_matrix.py — Varre uma pasta com arquivos Bruker OPUS e gera um Excel "matricial":
- Uma linha por amostra (arquivo)
- Colunas = eixo espectral comum (número de onda cm^-1, por padrão)
- Primeira coluna = sample_id (nome do arquivo, sem extensão)
- Abas: "matrix" (espectros alinhados) e "meta" (metadados por amostra)

Uso:
  python opus_folder_to_matrix.py /caminho/para/pasta -o saida.xlsx

Dependências (instale ao menos uma das leituras):
  pip install opusFC pandas openpyxl
  # ou
  pip install "spectrochempy[full]" pandas openpyxl

Observações:
- Alinhamento do eixo: usa o primeiro arquivo válido como referência.
- Se os eixos diferirem, interpola os demais para o eixo de referência.
- Se o eixo estiver em ordem decrescente, o script cuida disso ao interpolar.
- Metadados: tenta extrair tudo disponível e "achatar" em colunas.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def try_read_opus_opusFC(path: str):
    import opusFC
    data = opusFC.opus_reader(path)
    x = data['x']
    y = data['y']
    meta = {}
    for k, v in data.items():
        if k not in ('x', 'y'):
            if isinstance(v, dict):
                for kk, vv in v.items():
                    meta[f"{k}.{kk}"] = vv
            else:
                meta[k] = v
    return x, y, meta

def try_read_opus_spectrochempy(path: str):
    import spectrochempy as scp
    ds = scp.read_opus(path)
    if ds.ndim > 1:
        ds = ds[0, ...]
    x = ds.x.values.ravel().tolist()
    y = ds.values.ravel().tolist()
    meta = {}
    for k, v in ds.meta.items():
        try:
            meta[k] = str(v)
        except Exception:
            meta[k] = repr(v)
    return x, y, meta

def read_opus_any(path: str):
    last = None
    for reader_name, reader in [
        ("opusFC", try_read_opus_opusFC),
        ("spectrochempy", try_read_opus_spectrochempy),
    ]:
        try:
            return reader(*[path]), reader_name  # return (x,y,meta), name
        except Exception as e:
            last = e
    raise RuntimeError(f"Falha ao ler {path}: {last}")

def is_increasing(seq):
    return len(seq) <= 1 or seq[0] < seq[-1]

def interp_to_axis(x_src, y_src, x_ref):
    import numpy as np
    xs = np.asarray(x_src, dtype=float)
    ys = np.asarray(y_src, dtype=float)
    xr = np.asarray(x_ref, dtype=float)

    # np.interp requer eixo crescente
    if not is_increasing(xs):
        xs = xs[::-1]
        ys = ys[::-1]
    if not is_increasing(xr):
        xr = xr[::-1]
        flip_back = True
    else:
        flip_back = False

    yi = np.interp(xr, xs, ys, left=float('nan'), right=float('nan'))
    if flip_back:
        yi = yi[::-1]
    return yi.tolist()

def scan_opus_files(folder: Path) -> List[Path]:
    exts = {".0", ".OPUS", ".opus", ".spc", ".sp"}
    files = []
    for p in folder.rglob("*"):
        if p.is_file():
            if p.suffix in exts or p.suffix == "":
                files.append(p)
    # Ordem estável por nome
    files.sort()
    return files

def sanitize_header(values) -> List[str]:
    # Gera cabeçalho de colunas a partir do eixo x (formata com no máx. 4 decimais)
    headers = []
    for v in values:
        try:
            fv = float(v)
            s = f"{fv:.4f}".rstrip("0").rstrip(".")
        except Exception:
            s = str(v)
        headers.append(s)
    # Evita duplicatas (caso arredondamento colida)
    seen = {}
    out = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            out.append(f"{h}#{seen[h]}")
        else:
            seen[h] = 0
            out.append(h)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Pasta com arquivos OPUS")
    ap.add_argument("-o", "--output", default="opus_matrix.xlsx", help="Arquivo XLSX de saída")
    ap.add_argument("--meta-cols", nargs="*", default=["Sample", "ID", "Operator", "Date", "Time"],
                    help="Sugestões de colunas de metadados para destacar na frente (se existirem)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        ap.error(f"Pasta inválida: {folder}")

    files = scan_opus_files(folder)
    if not files:
        print("Nenhum arquivo OPUS encontrado.", file=sys.stderr)
        sys.exit(1)

    print(f"Encontrados {len(files)} arquivos. Lendo...")

    rows = []           # cada item: (sample_id, y_alinhado, meta_dict, x_original)
    ref_axis = None
    ref_file = None
    meta_all_keys = set()

    for i, f in enumerate(files, 1):
        try:
            (x, y, meta), reader_name = read_opus_any(str(f))
        except Exception as e:
            print(f"[ERRO] {f.name}: {e}", file=sys.stderr)
            continue

        # sample_id: nome do arquivo sem extensão(s)
        sample_id = f.stem
        # escolhe eixo de referência no primeiro sucesso
        if ref_axis is None:
            ref_axis = list(x)
            ref_file = f.name
            print(f"→ Eixo de referência: {len(ref_axis)} pontos ({ref_file})")

        # interpola se necessário
        if len(x) != len(ref_axis) or any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(x, ref_axis)):
            y_aligned = interp_to_axis(x, y, ref_axis)
        else:
            y_aligned = list(y)

        # normaliza metadados para dict simples de str
        flat = {}
        for k, v in (meta or {}).items():
            try:
                flat[str(k)] = str(v)
            except Exception:
                flat[str(k)] = repr(v)
        meta_all_keys.update(flat.keys())

        rows.append((sample_id, y_aligned, flat))

    if not rows:
        print("Nenhum arquivo pôde ser lido com sucesso.", file=sys.stderr)
        sys.exit(2)

    # DataFrames
    import pandas as pd

    header_x = sanitize_header(ref_axis)
    # "matrix": primeira coluna = sample_id; demais = eixo
    mat_df = pd.DataFrame(
        [dict({"sample_id": sid}, **{hx: y[i] for i, hx in enumerate(header_x)}) for sid, y, _ in rows]
    )

    # opcional: destacar algumas metacolunas no início
    preferred = [c for c in args.meta_cols if c in meta_all_keys]
    other_keys = sorted(k for k in meta_all_keys if k not in preferred)

    meta_rows = []
    for sid, _, m in rows:
        row = {"sample_id": sid}
        for k in preferred + other_keys:
            row[k] = m.get(k, "")
        meta_rows.append(row)
    meta_df = pd.DataFrame(meta_rows, columns=["sample_id"] + preferred + other_keys)

    out = Path(args.output)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        mat_df.to_excel(writer, index=False, sheet_name="matrix")
        meta_df.to_excel(writer, index=False, sheet_name="meta")
        # Abas extras: eixo de referência + info
        pd.DataFrame({"x_ref": header_x}).to_excel(writer, index=False, sheet_name="x_ref")
        info = pd.DataFrame([
            {"key": "reference_file", "value": ref_file},
            {"key": "n_files_parsed", "value": len(rows)},
        ])
        info.to_excel(writer, index=False, sheet_name="info")

    print(f"[OK] Gerado: {out} (matrix/meta/x_ref/info)")

if __name__ == "__main__":
    main()
