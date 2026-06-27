import streamlit as st

def metric_card(title: str, meaning: str, why: str, formula: str) -> None:
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(f"**📌 ¿Qué significa?**  \n{meaning}")
        st.markdown(f"**🎯 ¿Por qué es necesaria?**  \n{why}")
        st.markdown("**📐 Fórmula:**")
        st.latex(formula)

def render_all_metrics() -> None:
    st.markdown("### 📊 Métricas de Evaluación")
    
    # Primera fila - 2 columnas
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        metric_card(
            title="🎯 Recall (Sensibilidad)",
            meaning=(
                'De todo lo que verdaderamente era positivo **"Fraude"**, '
                "cuánto pudo el modelo clasificar como tal."
            ),
            why=(
                "En detección de fraude la clase minoritaria representa pérdidas directas. "
                "Un **Recall alto** minimiza falsos negativos (fraudes no detectados)."
            ),
            formula=r"Recall = \frac{TP}{TP + FN}",
        )
    
    with col2:
        metric_card(
            title="📊 Macro Precisión",
            meaning=(
                'De todo lo que el modelo predijo como **"Fraude"**, '
                "cuánto verdaderamente fue clasificado como tal (promedio macro por clase)."
            ),
            why=(
                "Evita generar demasiadas alertas falsas que saturan operaciones y "
                "generan costos operativos innecesarios."
            ),
            formula=r"Macro\ Precision = \frac{1}{n}\sum_{i=1}^{n}\frac{TP_i}{TP_i + FP_i}",
        )
    
    # Segunda fila - 2 columnas
    col3, col4 = st.columns(2, gap="medium")
    
    with col3:
        metric_card(
            title="⚖️ Macro F1-Score",
            meaning="Busca armonía entre precisión y recall, penalizando desequilibrios.",
            why=(
                "Ofrece una métrica global equilibrada cuando se quiere valorar por igual "
                "el rendimiento en todas las clases pese al desbalance."
            ),
            formula=r"Macro\ F1 = \frac{1}{n}\sum_{i=1}^{n}\frac{2 \cdot Precision_i \cdot Recall_i}{Precision_i + Recall_i}",
        )
    
    with col4:
        metric_card(
            title="📈 Balanced Accuracy",
            meaning=(
                "Mide qué tan bien reconoce el modelo cada clase, "
                "considerando el desbalance de la clase minoritaria **Fraude**."
            ),
            why=(
                "El accuracy tradicional puede ser engañoso con datos desbalanceados; "
                "esta métrica trata cada clase con igual peso."
            ),
            formula=r"Balanced\ Accuracy = \frac{1}{n}\sum_{i=1}^{n} Recall_i",
        )
    
    st.markdown("---")
    
    # Matriz de confusión - Ocupa todo el ancho
    st.markdown("### 🎯 Matriz de Confusión")
    with st.container(border=True):
        col5, col6 = st.columns([1, 1], gap="medium")
        
        with col5:
            st.markdown(
                """
                **📌 ¿Qué significa?**  
                Tabla que muestra **TP, TN, FP y FN** para evaluar 
                visualmente el comportamiento del clasificador.
                """
            )
        
        with col6:
            st.markdown(
                """
                **🎯 ¿Por qué es necesaria?**  
                Permite identificar si el modelo falla más 
                por no detectar fraudes **(FN)** o por bloquear 
                transacciones legítimas **(FP)**.
                """
            )
        
        st.markdown("**📐 Representación:**")
        st.latex(r"Matriz\ de\ Confusión = \begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}")
        
        # Ejemplo visual de la matriz
        st.markdown("**Ejemplo de interpretación:**")
        col7, col8, col9, col10 = st.columns(4)
        with col7:
            st.metric("✅ TN", "Verdaderos Negativos")
        with col8:
            st.metric("❌ FP", "Falsos Positivos")
        with col9:
            st.metric("❌ FN", "Falsos Negativos")
        with col10:
            st.metric("✅ TP", "Verdaderos Positivos")