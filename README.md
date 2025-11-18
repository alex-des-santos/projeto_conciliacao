# Streamlit — Dashboard de Conciliação Bancária

Descrição
O repositório contém uma UI Streamlit que reaproveita o motor de conciliação e os dados do projeto original para explorar resultados de conciliação bancária de forma interativa. É uma implementação leve para visualização, filtro e inspeção de dados mock/CSV sem alterar a lógica de negócio existente.

O que está aqui
- streamlit/app.py — aplicação Streamlit (entrada).
- streamlit/requirements.txt — dependências da UI (Streamlit, Plotly, etc.).
- reconciliation_engine.py — lógica de conciliação reaproveitada.
- config.py — configurações usadas pelo motor e pela UI.
- data/*.csv — arquivos CSV usados como fonte de dados (mock/original).
- Observação: mantém a compatibilidade com o projeto Dash original, sem duplicar a lógica.

Principais funcionalidades
- Visualização por abas: conciliações, pendências TotalBank, pendências SAP.
- Filtros interativos: situação, mês, período.
- Gráficos Plotly compostos (barras + linha de SLA).
- Layout limpo com cards e ênfase na usabilidade para análise exploratória.

Como executar (recomendado)
1. Crie e ative um ambiente virtual (não instale no Python global):
   - Windows:
     python -m venv .venv
     .venv\Scripts\activate
   - macOS/Linux:
     python3 -m venv .venv
     source .venv/bin/activate
2. Instale dependências:
   pip install -r streamlit/requirements.txt
3. Rode o app:
   streamlit run streamlit/app.py
4. Acesse: http://localhost:8501

Observações rápidas
- A lógica de conciliação permanece no mesmo módulo para evitar discrepâncias entre UIs.
- Se quiser testar com outro conjunto de dados, troque/aponte os CSVs em config.py.
- Projetado para inspeção e prototipagem — ajuste performance e segurança antes de produção.
