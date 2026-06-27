import streamlit as st
from componentes.metrics_cards import render_all_metrics

def main():
    st.set_page_config(
        page_title="Fashion MNIST - Clasificación de Prendas",
        page_icon="👕",
        layout="wide"
    )
    
    st.title("Proyecto de Clasificación de Prendas con Deep Learning")
    st.markdown("---")
    
    # 1.1 Problemática
    st.header("1.1 Problemática")
    st.markdown(
        """
        La clasificación automática de prendas de ropa es un problema fundamental en el comercio 
        electrónico, la logística y la visión por computadora. Las empresas necesitan sistemas 
        automáticos que puedan identificar y categorizar productos de vestimenta a partir de 
        imágenes para optimizar inventarios, mejorar la experiencia de usuario y automatizar 
        procesos de clasificación.
        
        En este proyecto, abordamos este desafío construyendo un pipeline completo de Deep 
        Learning para clasificar imágenes de prendas en 10 categorías diferentes, utilizando 
        el dataset Fashion MNIST.
        """
    )
    
    # 1.2 Contexto
    st.header("1.2 Contexto")
    st.markdown(
        """
        La clasificación automática de prendas de ropa es un problema real en e-commerce, logística y 
        visión por computadora. En este datatón construirán un pipeline completo de Deep Learning — 
        desde la exploración del dataset hasta una aplicación interactiva desplegada— usando el dataset 
        Fashion MNIST.
        
        El foco de esta sesión es doble: entender las diferencias entre arquitecturas (MLP vs CNN) y saber 
        comunicar resultados a través de una interfaz de usuario construida con Streamlit.
        """
    )
    
    st.markdown("---")
    
    # 1.3 Objetivos del proyecto
    st.header("1.3 Objetivos del proyecto")
    st.subheader("Objetivo general")
    st.markdown(
        """
        Desarrollar y comparar diferentes arquitecturas de Deep Learning (MLP, CNN y CNN con 
        técnicas de regularización) para la clasificación de prendas de vestir utilizando el 
        dataset Fashion MNIST, con el fin de identificar la arquitectura más efectiva para 
        este problema de visión por computadora.
        """
    )
        
    st.markdown("---")
    
    # Dataset
    st.header("📊 Dataset: Fashion MNIST")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            **Fuente**: Disponible directamente desde `tensorflow.keras.datasets` (sin descarga manual)
            
            **Tarea**: Clasificación multiclase → 10 categorías de ropa
            
            **Tamaño**: 70,000 imágenes en escala de grises de 28×28 píxeles (60,000 train / 10,000 test)
            
            **Carga del dataset**:
            ```python
            from tensorflow.keras.datasets import fashion_mnist
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
            """
        )
    with col2:
        st.markdown("Clases disponibles:")
        clases = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }
        for id_clase, nombre in clases.items():
            st.write(f"{id_clase}. {nombre}")
    
    st.markdown("---")
    
    # Flujo de trabajo
    st.header("🔄 Flujo de trabajo")
    flujo = {
        "1. Exploración y Preparación de Datos (EDA)": [
            "Cargar el dataset con fashion_mnist.load_data()",
            "Mostrar la forma (shape) de los conjuntos de entrenamiento y test",
            "Visualizar al menos 2 imágenes por clase con su etiqueta correspondiente",
            "Graficar la distribución de clases y comentar si existe desbalance",
            "Normalizar los píxeles al rango [0, 1] y explicar por qué es importante",
            "Reshape para MLP (aplanar) y para CNN (agregar canal)"
        ],
        "2. Modelo 1 – MLP (Baseline)": [
            "Aplanar las imágenes con Flatten para alimentar el MLP",
            "Definir y documentar la arquitectura (número de capas, neuronas, activaciones)",
            "Compilar con optimizer, loss (sparse_categorical_crossentropy) y métrica adecuados",
            "Entrenar el modelo (mínimo 10 épocas)",
            "Graficar curvas de loss y accuracy (train vs validación)",
            "Evaluar en el conjunto de test"
        ],
        "3. Modelo 2 – CNN": [
            "Definir arquitectura CNN con al menos: 2 bloques Conv2D + MaxPooling2D",
            "Capa Flatten o GlobalAveragePooling2D",
            "Capas Dense finales con activación softmax",
            "Explicar qué detecta una capa convolucional en imágenes de ropa",
            "Entrenar el modelo (mínimo 10 épocas)",
            "Graficar curvas de entrenamiento",
            "Evaluar en test y comparar con MLP"
        ],
        "4. Modelo 3 – Arquitectura Investigada": [
            "Aplicar al menos 2 técnicas de regularización:",
            "• Dropout: Desactiva neuronas aleatoriamente",
            "• BatchNormalization: Normaliza activaciones entre capas",
            "• Data Augmentation: Rotaciones, flips, zoom",
            "• L2 Regularization: Penaliza pesos grandes",
            "Indicar qué técnicas eligieron y justificar la elección",
            "Implementar sobre la arquitectura CNN",
            "Entrenar y graficar curvas de entrenamiento",
            "Evaluar en test y comparar con modelos anteriores"
        ],
        "5. Evaluación y Métricas": [
            "Calcular métricas usando sklearn",
            "Graficar la Matriz de Confusión del mejor modelo",
            "Identificar cuáles clases se confunden con más frecuencia",
            "Discutir cuál modelo generaliza mejor y por qué"
        ]
    }

    for etapa, pasos in flujo.items():
        with st.expander(etapa):
            for paso in pasos:
                st.write(f"• {paso}")

    st.markdown("---")
    
    # Botón para ir a la simulación - Usando st.page_link
    st.header("🚀 Simulación del Modelo")
    st.markdown("Visualiza el rendimiento del modelo entrenado con métricas y predicciones en tiempo real")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        # Usar st.page_link en lugar de st.switch_page
        st.page_link("pages/simulacion.py", label="🔬 Ir a la Simulación", use_container_width=True)
    
    st.markdown("---")

if __name__ == "__main__":
    main()