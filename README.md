# Streamlit Dashboard

Replica do painel de conciliacao bancaria em Streamlit mantendo o motor e os dados do projeto original.

## Como executar

1. Garanta que o ambiente virtual (.venv) usado no projeto principal esteja ativo.
2. Instale as dependencias especificas do Streamlit:
   ```bash
   pip install -r streamlit/requirements.txt
   ```
3. Rode o app:
   ```bash
   streamlit run streamlit/app.py
   ```
4. O dashboard abre em http://localhost:8501 por padrao.

## Diferenciais visuais
- Layout branco com cards em relevo, inspirado na imagem enviada.
- Filtros de situacao, mes e periodo para explorar rapidamente os dados mock.
- Grafico Plotly que combina barras (tempo medio por descricao) e linha de referencia do SLA.
- Abas para conciliações, pendencias do TotalBank e pendencias do SAP.

O app reaproveita `reconciliation_engine.py`, `config.py` e os CSVs originais, preservando o projeto Dash existente.
