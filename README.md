# Introdução

O utilitário **opus_folder_to_matrix.py** converte pastas com arquivos Bruker
OPUS em matrizes tabulares prontas para análise em ferramentas como Excel,
Python ou R. Cada amostra vira uma linha, com o eixo espectral alinhado e os
metadados achatados em colunas. A saída pode ser gerada em formato XLSX (com
aba para matriz, metadados, eixo de referência e informações) ou em múltiplos
arquivos CSV equivalentes.

# Instalação

Requisitos mínimos:

- Python 3.8 ou superior.
- `pip` para gerenciar pacotes.
- Pelo menos um leitor OPUS compatível: `opusFC` ou `spectrochempy`.
- `pandas` para manipulação tabular.
- `openpyxl` (somente necessário para exportar XLSX).

Instalação rápida em um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install pandas openpyxl opusFC
# OU substitua a última linha por:
# pip install pandas openpyxl "spectrochempy[full]"
```

# Uso

O script aceita como argumento principal a pasta que contém os arquivos OPUS.
Alguns exemplos de execução:

```bash
# Gera um XLSX com todas as abas padrão
python opus_folder_to_matrix.py dados/ftir --format xlsx -o matriz.xlsx

# Cria arquivos CSV separados e prioriza colunas de metadados customizadas
python opus_folder_to_matrix.py dados/ftir --format csv --meta-cols Sample ID Operator Date Time

# Força o uso do leitor opusFC e habilita mensagens informativas
python opus_folder_to_matrix.py dados/ftir --reader opusFC --verbose -o resultados.xlsx
```

Quando `--format=csv` for escolhido, o nome informado em `--output` servirá de
base para quatro arquivos: `*_matrix.csv`, `*_meta.csv`, `*_x_ref.csv` e
`*_info.csv`.

# Opções

| Opção | Descrição |
| --- | --- |
| `folder` | Caminho da pasta com arquivos OPUS. |
| `-o`, `--output` | Nome do arquivo de saída. Caso omitido, usa `opus_matrix.<formato>`. |
| `--format {xlsx,csv}` | Define o formato de exportação (padrão: `xlsx`). |
| `--meta-cols` | Lista opcional de colunas de metadados para destacar primeiro nas planilhas. |
| `--reader {opusFC,spectrochempy}` | Força a utilização de um leitor específico e valida se a dependência correspondente está instalada. |
| `--verbose` | Configura o logger para exibir mensagens informativas (nível INFO). |
| `--quiet` | Restringe a saída a mensagens de erro (nível ERROR). |

# Compilação

Para gerar um executável simples com [PyInstaller](https://pyinstaller.org/):

```bash
pip install pyinstaller
pyinstaller opus_folder_to_matrix.py
# O binário resultante ficará em dist/opus_folder_to_matrix/
```

Para empacotar a interface gráfica basta apontar o PyInstaller para o módulo
da GUI:

```bash
pyinstaller --onefile opus_folder_to_matrix_gui.py
```

Como ``opus_folder_to_matrix_gui.py`` importa diretamente o módulo de
conversão, não é necessário utilizar opções extras como ``--add-data`` para que
o script principal seja incluído no executável.

# Licença

Este projeto não possui uma licença explícita. Entre em contato com os
responsáveis antes de redistribuir ou reutilizar o código.

