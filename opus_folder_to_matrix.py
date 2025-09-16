#!/usr/bin/env python3
"""
opus_folder_to_matrix.py — Varre uma pasta com arquivos Bruker OPUS e gera uma
matriz tabular (XLSX ou CSV):

- Uma linha por amostra (arquivo)
- Colunas = eixo espectral comum (número de onda cm^-1, por padrão)
- Primeira coluna = ``sample_id`` (nome do arquivo, sem extensão)
- Saída XLSX: abas ``matrix``/``meta``/``x_ref``/``info``
- Saída CSV: arquivos ``*_matrix.csv``/``*_meta.csv``/``*_x_ref.csv``/``*_info.csv``

Uso básico:
  python opus_folder_to_matrix.py /caminho/para/pasta --format xlsx -o matriz.xlsx
  python opus_folder_to_matrix.py /caminho/para/pasta --format csv -o matriz.csv

Dependências principais (instale ao menos uma das leituras):
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
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


logger = logging.getLogger(__name__)


ReaderResult = Tuple[List[float], List[float], Dict[str, Any]]
ReaderFunc = Callable[[str], ReaderResult]


def try_read_opus_opusFC(path: str) -> ReaderResult:
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


def try_read_opus_spectrochempy(path: str) -> ReaderResult:
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


READERS: Dict[str, ReaderFunc] = {
    "opusFC": try_read_opus_opusFC,
    "spectrochempy": try_read_opus_spectrochempy,
}

def read_opus_any(path: str, reader_names: Optional[Sequence[str]] = None) -> Tuple[ReaderResult, str]:
    """Tenta ler um arquivo OPUS com qualquer leitor suportado.

    Args:
        path: Caminho do arquivo OPUS a ser processado.
        reader_names: Sequência opcional com os nomes dos leitores a tentar,
            na ordem desejada.

    Returns:
        Uma tupla contendo o resultado do leitor escolhido (``x``, ``y`` e
        metadados) e o nome da biblioteca utilizada.

    Raises:
        RuntimeError: Se todos os leitores falharem ao interpretar o arquivo.
    """
    last: Optional[Exception] = None
    names: Sequence[str] = reader_names or tuple(READERS.keys())
    for reader_name in names:
        reader = READERS[reader_name]
        try:
            return reader(path), reader_name  # return (x,y,meta), name
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
    indicada e gera uma matriz tabular (XLSX ou CSV) com espectros e
    metadados.

    Raises:
        SystemExit: Com código diferente de zero quando a pasta é inválida ou
            nenhum arquivo é processado com sucesso.
        RuntimeError: Se nenhum eixo de referência puder ser determinado.
    """

    ap = argparse.ArgumentParser(
        description="Converte uma pasta com arquivos Bruker OPUS em planilhas tabulares."
    )
    ap.add_argument("folder", help="Pasta com arquivos OPUS")
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Arquivo de saída (padrão: opus_matrix.<formato>)",
    )
    ap.add_argument(
        "--format",
        choices=("xlsx", "csv"),
        default="xlsx",
        help="Formato do arquivo de saída. Padrão: xlsx.",
    )
    log_group = ap.add_mutually_exclusive_group()
    log_group.add_argument(
        "--verbose",
        action="store_true",
        help="Exibe mensagens informativas (nível INFO).",
    )
    log_group.add_argument(
        "--quiet",
        action="store_true",
        help="Mostra apenas mensagens de erro (nível ERROR).",
    )
    ap.add_argument(
        "--reader",
        choices=tuple(READERS.keys()),
        help="Força o uso de um leitor específico (opusFC ou spectrochempy).",
    )
    ap.add_argument(
        "--meta-cols",
        nargs="*",
        default=["Sample", "ID", "Operator", "Date", "Time"],
        help="Sugestões de colunas de metadados para destacar na frente (se existirem)",
    )
    args = ap.parse_args()

    if args.verbose:
        log_level = logging.INFO
    elif args.quiet:
        log_level = logging.ERROR
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s", force=True)
    logger.setLevel(log_level)

    if args.reader:
        module_name = "opusFC" if args.reader == "opusFC" else "spectrochempy"
        try:
            importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - depende de pacotes externos
            suggestion = "pip install opusFC" if module_name == "opusFC" else 'pip install "spectrochempy[full]"'
            ap.error(
                f"O leitor '{args.reader}' foi selecionado, mas o pacote '{module_name}' não está disponível. "
                f"Instale-o com `{suggestion}`."
            )
        readers_to_try: Sequence[str] = [args.reader]
    else:
        readers_to_try = list(READERS.keys())

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        ap.error(f"Pasta inválida: {folder}")

    files: List[Path] = scan_opus_files(folder)
    if not files:
        logger.error("Nenhum arquivo OPUS encontrado.")
        sys.exit(1)

    logger.info("Encontrados %s arquivos. Lendo...", len(files))

    rows: List[Tuple[str, List[float], Dict[str, str]]] = []
    ref_axis: Optional[List[float]] = None
    ref_file: Optional[str] = None
    meta_all_keys: Set[str] = set()

    for f in files:
        try:
            (x, y, meta), _reader_name = read_opus_any(str(f), readers_to_try)
        except Exception as e:  # pragma: no cover - depende de dados externos
            logger.error("Falha ao ler %s: %s", f.name, e)
            continue

        sample_id: str = f.stem
        if ref_axis is None:
            ref_axis = list(x)
            ref_file = f.name
            logger.info("→ Eixo de referência: %s pontos (%s)", len(ref_axis), ref_file)

        if len(x) != len(ref_axis) or any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(x, ref_axis)):
            y_aligned: List[float] = interp_to_axis(x, y, ref_axis)
        else:
            y_aligned = list(y)

        flat: Dict[str, str] = {}
        for k, v in (meta or {}).items():
            try:
                flat[str(k)] = str(v)
            except Exception:
                flat[str(k)] = repr(v)
        meta_all_keys.update(flat.keys())

        rows.append((sample_id, y_aligned, flat))

    if not rows:
        logger.error("Nenhum arquivo pôde ser lido com sucesso.")
        sys.exit(2)

    if ref_axis is None:
        raise RuntimeError("Eixo de referência não definido após o processamento dos arquivos.")

    import pandas as pd

    header_x: List[str] = sanitize_header(ref_axis)
    mat_df = pd.DataFrame(
        [dict({"sample_id": sid}, **{hx: y[i] for i, hx in enumerate(header_x)}) for sid, y, _ in rows]
    )

    preferred: List[str] = [c for c in args.meta_cols if c in meta_all_keys]
    other_keys: List[str] = sorted(k for k in meta_all_keys if k not in preferred)

    meta_rows: List[Dict[str, str]] = []
    for sid, _, m in rows:
        row: Dict[str, str] = {"sample_id": sid}
        for k in preferred + other_keys:
            row[k] = m.get(k, "")
        meta_rows.append(row)
    meta_df = pd.DataFrame(meta_rows, columns=["sample_id"] + preferred + other_keys)

    info_df = pd.DataFrame(
        [
            {"key": "reference_file", "value": ref_file},
            {"key": "n_files_parsed", "value": len(rows)},
        ]
    )
    x_ref_df = pd.DataFrame({"x_ref": header_x})

    if args.output is None:
        out = Path(f"opus_matrix.{args.format}")
    else:
        out = Path(args.output)

    if out.suffix.lower() != f".{args.format}":
        logger.info("Ajustando extensão do arquivo de saída para .%s", args.format)
        out = out.with_suffix(f".{args.format}")

    if args.format == "xlsx":
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            mat_df.to_excel(writer, index=False, sheet_name="matrix")
            meta_df.to_excel(writer, index=False, sheet_name="meta")
            x_ref_df.to_excel(writer, index=False, sheet_name="x_ref")
            info_df.to_excel(writer, index=False, sheet_name="info")
        logger.info("Arquivo XLSX gerado: %s (abas matrix/meta/x_ref/info)", out)
    else:
        matrix_path = out
        meta_path = out.with_name(f"{out.stem}_meta.csv")
        xref_path = out.with_name(f"{out.stem}_x_ref.csv")
        info_path = out.with_name(f"{out.stem}_info.csv")
        mat_df.to_csv(matrix_path, index=False)
        meta_df.to_csv(meta_path, index=False)
        x_ref_df.to_csv(xref_path, index=False)
        info_df.to_csv(info_path, index=False)
        logger.info(
            "Arquivos CSV gerados: %s, %s, %s, %s",
            matrix_path,
            meta_path,
            xref_path,
            info_path,
        )

if __name__ == "__main__":
    main()
