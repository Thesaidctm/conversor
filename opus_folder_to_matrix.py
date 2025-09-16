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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def try_read_opus_opusFC(path: str) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """Lê um arquivo OPUS usando ``opusFC``.

    Args:
        path: Caminho absoluto ou relativo do arquivo OPUS.

    Returns:
        Uma tupla contendo o eixo ``x`` convertido em ``list`` de ``float``, os
        valores ``y`` também em ``list`` de ``float`` e um dicionário de
        metadados achatados.

    Raises:
        ImportError: Se o pacote ``opusFC`` não estiver instalado.
        ValueError: Se os dados ``x`` ou ``y`` não puderem ser convertidos para ``float``.
        TypeError: Se os dados ``x`` ou ``y`` não forem iteráveis numéricos.
        Exception: Propaga quaisquer erros de leitura originados de ``opusFC``.
    """
    import opusFC
    data: Dict[str, Any] = opusFC.opus_reader(path)
    x_values: List[float] = [float(value) for value in data["x"]]
    y_values: List[float] = [float(value) for value in data["y"]]
    meta: Dict[str, Any] = {}
    for k, v in data.items():
        if k not in ('x', 'y'):
            if isinstance(v, dict):
                for kk, vv in v.items():
                    meta[f"{k}.{kk}"] = vv
            else:
                meta[k] = v
    return x_values, y_values, meta


def try_read_opus_spectrochempy(path: str) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """Lê um arquivo OPUS usando ``spectrochempy``.

    Args:
        path: Caminho absoluto ou relativo do arquivo OPUS.

    Returns:
        Uma tupla com o eixo ``x`` como ``list`` de ``float``, os valores ``y``
        como ``list`` de ``float`` e um dicionário com os metadados convertidos
        para ``str``.

    Raises:
        ImportError: Se ``spectrochempy`` não estiver disponível.
        Exception: Propaga erros de leitura lançados por ``spectrochempy``.
    """
    import spectrochempy as scp
    ds = scp.read_opus(path)
    if ds.ndim > 1:
        ds = ds[0, ...]
    x: List[float] = ds.x.values.ravel().tolist()
    y: List[float] = ds.values.ravel().tolist()
    meta: Dict[str, Any] = {}
    for k, v in ds.meta.items():
        try:
            meta[k] = str(v)
        except Exception:
            meta[k] = repr(v)
    return x, y, meta

def read_opus_any(path: str) -> Tuple[Tuple[List[float], List[float], Dict[str, Any]], str]:
    """Tenta ler um arquivo OPUS com qualquer leitor suportado.

    Args:
        path: Caminho do arquivo OPUS a ser processado.

    Returns:
        Uma tupla contendo o resultado do leitor escolhido (``x``, ``y`` e
        metadados) e o nome da biblioteca utilizada.

    Raises:
        RuntimeError: Se todos os leitores falharem ao interpretar o arquivo.
    """
    last: Optional[Exception] = None
    for reader_name, reader in [
        ("opusFC", try_read_opus_opusFC),
        ("spectrochempy", try_read_opus_spectrochempy),
    ]:
        try:
            return reader(*[path]), reader_name  # return (x,y,meta), name
        except Exception as e:
            last = e
    raise RuntimeError(f"Falha ao ler {path}: {last}")

def is_increasing(seq: Sequence[float]) -> bool:
    """Verifica se uma sequência está em ordem crescente.

    Args:
        seq: Sequência numérica a ser avaliada.

    Returns:
        ``True`` se o primeiro elemento for menor que o último ou se houver
        zero/um elemento; ``False`` caso contrário.

    Raises:
        TypeError: Se os elementos da sequência não forem comparáveis.
    """
    return len(seq) <= 1 or seq[0] < seq[-1]


def interp_to_axis(
    x_src: Sequence[float],
    y_src: Sequence[float],
    x_ref: Sequence[float],
) -> List[float]:
    """Interpola um espectro para um eixo de referência.

    Args:
        x_src: Eixo original do espectro de entrada.
        y_src: Valores do espectro correspondentes a ``x_src``.
        x_ref: Eixo alvo para o qual os dados serão interpolados.

    Returns:
        Uma lista de valores ``y`` alinhados ao eixo ``x_ref``.

    Raises:
        ValueError: Se as sequências não puderem ser convertidas para float ou
            se possuírem comprimentos incompatíveis para interpolação.
    """
    import numpy as np
    xs: np.ndarray = np.asarray(x_src, dtype=float)
    ys: np.ndarray = np.asarray(y_src, dtype=float)
    xr: np.ndarray = np.asarray(x_ref, dtype=float)

    # np.interp requer eixo crescente
    if not is_increasing(xs):
        xs = xs[::-1]
        ys = ys[::-1]
    if not is_increasing(xr):
        xr = xr[::-1]
        flip_back: bool = True
    else:
        flip_back = False

    yi: np.ndarray = np.interp(xr, xs, ys, left=float('nan'), right=float('nan'))
    if flip_back:
        yi = yi[::-1]
    return yi.tolist()

def scan_opus_files(folder: Path) -> List[Path]:
    """Varre uma pasta em busca de arquivos compatíveis com OPUS.

    Args:
        folder: Diretório raiz a ser explorado.

    Returns:
        Lista ordenada de caminhos para arquivos com extensões reconhecidas
        como dados OPUS.

    Raises:
        FileNotFoundError: Se ``folder`` não existir.
        NotADirectoryError: Se ``folder`` não apontar para um diretório.
    """
    if not folder.exists():
        raise FileNotFoundError(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    exts: Set[str] = {".0", ".OPUS", ".opus", ".spc", ".sp"}
    files: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file():
            if p.suffix in exts or p.suffix == "":
                files.append(p)
    # Ordem estável por nome
    files.sort()
    return files

def sanitize_header(values: Sequence[Any]) -> List[str]:
    """Formata rótulos de coluna a partir do eixo ``x`` de referência.

    Args:
        values: Sequência original do eixo ``x``.

    Returns:
        Lista de strings pronta para uso como cabeçalho de colunas, com até
        quatro casas decimais e sufixos para evitar duplicatas.

    Raises:
        TypeError: Propagada se ``values`` não for uma sequência.
        Exception: Propagada se algum elemento não puder ser convertido em
            ``str``.
    """
    # Gera cabeçalho de colunas a partir do eixo x (formata com no máx. 4 decimais)
    headers: List[str] = []
    for v in values:
        try:
            fv = float(v)
            s = f"{fv:.4f}".rstrip("0").rstrip(".")
        except Exception:
            s = str(v)
        headers.append(s)
    # Evita duplicatas (caso arredondamento colida)
    seen: Dict[str, int] = {}
    out: List[str] = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            out.append(f"{h}#{seen[h]}")
        else:
            seen[h] = 0
            out.append(h)
    return out

def main() -> None:
    """Ponto de entrada do utilitário de conversão de arquivos OPUS.

    Configura os argumentos da linha de comando, processa os arquivos da pasta
    indicada e gera um arquivo XLSX com matriz de espectros e metadados.

    Raises:
        SystemExit: Com código diferente de zero quando a pasta é inválida ou
            nenhum arquivo é processado com sucesso.
        RuntimeError: Se nenhum eixo de referência puder ser determinado.
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Pasta com arquivos OPUS")
    ap.add_argument("-o", "--output", default="opus_matrix.xlsx", help="Arquivo XLSX de saída")
    ap.add_argument("--meta-cols", nargs="*", default=["Sample", "ID", "Operator", "Date", "Time"],
                    help="Sugestões de colunas de metadados para destacar na frente (se existirem)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        ap.error(f"Pasta inválida: {folder}")

    files: List[Path] = scan_opus_files(folder)
    if not files:
        print("Nenhum arquivo OPUS encontrado.", file=sys.stderr)
        sys.exit(1)

    print(f"Encontrados {len(files)} arquivos. Lendo...")

    rows: List[Tuple[str, List[float], Dict[str, str]]] = []
    ref_axis: Optional[List[float]] = None
    ref_file: Optional[str] = None
    meta_all_keys: Set[str] = set()

    for f in files:
        try:
            (x, y, meta), _reader_name = read_opus_any(str(f))
        except Exception as e:
            print(f"[ERRO] {f.name}: {e}", file=sys.stderr)
            continue

        # sample_id: nome do arquivo sem extensão(s)
        sample_id: str = f.stem
        # escolhe eixo de referência no primeiro sucesso
        if ref_axis is None:
            ref_axis = list(x)
            ref_file = f.name
            print(f"→ Eixo de referência: {len(ref_axis)} pontos ({ref_file})")

        # interpola se necessário
        if len(x) != len(ref_axis) or any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(x, ref_axis)):
            y_aligned: List[float] = interp_to_axis(x, y, ref_axis)
        else:
            y_aligned = list(y)

        # normaliza metadados para dict simples de str
        flat: Dict[str, str] = {}
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

    if ref_axis is None:
        raise RuntimeError("Eixo de referência não definido após o processamento dos arquivos.")
    assert ref_axis is not None

    # DataFrames
    import pandas as pd

    header_x: List[str] = sanitize_header(ref_axis)
    # "matrix": primeira coluna = sample_id; demais = eixo
    mat_df = pd.DataFrame(
        [dict({"sample_id": sid}, **{hx: y[i] for i, hx in enumerate(header_x)}) for sid, y, _ in rows]
    )

    # opcional: destacar algumas metacolunas no início
    preferred: List[str] = [c for c in args.meta_cols if c in meta_all_keys]
    other_keys: List[str] = sorted(k for k in meta_all_keys if k not in preferred)

    meta_rows: List[Dict[str, str]] = []
    for sid, _, m in rows:
        row: Dict[str, str] = {"sample_id": sid}
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
