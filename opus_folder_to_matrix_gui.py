#!/usr/bin/env python3
"""Interface gráfica simples para ``opus_folder_to_matrix.py``.

A aplicação pede ao usuário uma pasta com arquivos OPUS, o formato de saída
(XLSX ou CSV) e o caminho do arquivo gerado, executando o script em modo linha
de comando em seguida.

Compilação (opcional) com PyInstaller::

    pyinstaller --onefile opus_folder_to_matrix_gui.py

O módulo ``opus_folder_to_matrix`` é importado diretamente pela interface, não
havendo necessidade de parâmetros extras como ``--add-data``.
"""

from __future__ import annotations

import io
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from opus_folder_to_matrix import run as run_conversion

FORMATS = ("xlsx", "csv")
APP_TITLE = "Conversor OPUS para matriz"


def converter(formato_var: tk.StringVar) -> None:
    """Abre diálogos para coletar entradas e executa a conversão."""

    pasta = filedialog.askdirectory(title="Selecione a pasta com arquivos OPUS")
    if not pasta:
        return

    formato = formato_var.get() or FORMATS[0]
    default_ext = ".xlsx" if formato == "xlsx" else ".csv"
    filetypes = (
        [("Planilha Excel (*.xlsx)", "*.xlsx")] if formato == "xlsx" else [("CSV (*.csv)", "*.csv")]
    )

    saida = filedialog.asksaveasfilename(
        title="Escolha o arquivo de saída",
        defaultextension=default_ext,
        filetypes=filetypes,
        initialfile=f"matriz{default_ext}",
    )
    if not saida:
        return

    log_buffer = io.StringIO()
    try:
        outputs = run_conversion(
            folder=pasta,
            output=saida,
            fmt=formato,
            log_stream=log_buffer,
        )
    except Exception as exc:
        logs = log_buffer.getvalue().strip()
        message_lines = ["Falha ao converter os arquivos OPUS.", str(exc)]
        if logs:
            message_lines.extend(["", logs])
        messagebox.showerror("Erro na conversão", "\n".join(message_lines))
        return

    logs = log_buffer.getvalue().strip()
    output_paths = [str(path) for path in outputs] if outputs else [saida]
    message_lines = ["Conversão finalizada com sucesso!"]
    if len(output_paths) == 1:
        message_lines.append(f"Arquivo salvo em:\n{output_paths[0]}")
    else:
        message_lines.append("Arquivos gerados:")
        message_lines.extend(output_paths)
    if logs:
        message_lines.extend(["", logs])
    messagebox.showinfo("Conversão concluída", "\n".join(message_lines))


def build_gui() -> tk.Tk:
    """Cria e configura a janela principal da aplicação."""

    root = tk.Tk()
    root.title(APP_TITLE)
    root.resizable(False, False)

    main_frame = ttk.Frame(root, padding=20)
    main_frame.grid(row=0, column=0, sticky="nsew")
    main_frame.columnconfigure(0, weight=1)

    ttk.Label(
        main_frame,
        text="Selecione o formato desejado e clique em \"Converter\" para escolher as pastas.",
        wraplength=320,
        justify="center",
    ).grid(row=0, column=0, pady=(0, 12))

    formato_var = tk.StringVar(value=FORMATS[0])
    ttk.Label(main_frame, text="Formato de saída:").grid(row=1, column=0, sticky="w")
    formato_combo = ttk.Combobox(
        main_frame,
        textvariable=formato_var,
        values=FORMATS,
        state="readonly",
        width=10,
    )
    formato_combo.grid(row=2, column=0, pady=(0, 12))
    formato_combo.current(0)

    ttk.Button(
        main_frame,
        text="Converter pasta OPUS…",
        command=lambda: converter(formato_var),
    ).grid(row=3, column=0)

    return root


def main() -> None:
    """Ponto de entrada da aplicação."""

    root = build_gui()
    root.mainloop()


if __name__ == "__main__":
    main()
