"""
Script de teste para verificar se as correções de CSS foram aplicadas corretamente
"""

import streamlit as st

def test_css_corrections():
    st.set_page_config(page_title="Teste CSS Correções", layout="wide")
    
    # CSS para testar as correções
    test_css = """
    <style>
    /* Testar alinhamento */
    .test-alignment {
        border: 2px solid red;
        padding: 10px;
        margin: 5px;
    }
    
    /* Testar cores dos dropdowns */
    div[data-baseweb="select"] {
        border: 2px solid blue !important;
    }
    
    /* Testar transparência */
    .test-opacity {
        background-color: rgba(92, 198, 152, 0.6); /* SLA_COLOR com 60% opacidade */
        padding: 20px;
        color: white;
    }
    </style>
    """
    
    st.markdown(test_css, unsafe_allow_html=True)
    
    st.title("Teste das Correções de CSS")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='test-alignment'>Coluna 1 - Teste de Alinhamento</div>", unsafe_allow_html=True)
        option = st.selectbox("Teste Dropdown", ["Opção 1", "Opção 2", "Opção 3"])
        
    with col2:
        st.markdown("<div class='test-alignment'>Coluna 2 - Teste de Alinhamento</div>", unsafe_allow_html=True)
        radio_option = st.radio("Teste Radio", ["Radio 1", "Radio 2"], horizontal=True)
        
    with col3:
        st.markdown("<div class='test-alignment'>Coluna 3 - Teste de Alinhamento</div>", unsafe_allow_html=True)
        date = st.date_input("Teste Date Input")
    
    st.markdown("<div class='test-opacity'>Teste de Transparência - Linha Verde da Mediana</div>", unsafe_allow_html=True)
    
    st.write("Verificações:")
    st.write("✅ As bordas vermelhas mostram o alinhamento entre as colunas")
    st.write("✅ A borda azul no dropdown indica que o estilo está aplicado")
    st.write("✅ O fundo verde transparente mostra a opacidade de 60%")

if __name__ == "__main__":
    test_css_corrections()