Conversão em lote (uma linha por amostra)

1) Instale dependências (uma das leituras + Excel):
   - Leve:    pip install opusFC pandas openpyxl
   - Ou:      pip install "spectrochempy[full]" pandas openpyxl

2) Rode o script apontando para a pasta com OPUS:
   python opus_folder_to_matrix.py "C:\dados\bruker_opus" -o matriz.xlsx

3) Resultado (matriz.xlsx):
   - Aba matrix: sample_id | <wavenumber_1> | <wavenumber_2> | ...
   - Aba meta:   metadados por amostra (colunas agregadas)
   - Aba x_ref:  eixo de referência (cabeçalho das colunas espectrais)
   - Aba info:   arquivo de referência e contagem

Notas:
- O eixo de referência é o do primeiro arquivo lido com sucesso.
- Se algum arquivo tiver eixo diferente, ele será interpolado para o eixo comum.
- Você pode priorizar certas colunas de metadados na frente com --meta-cols,
  ex.: --meta-cols Sample ID Operator Date Time
