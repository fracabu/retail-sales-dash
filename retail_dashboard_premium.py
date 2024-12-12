import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Configurazione della pagina
st.set_page_config(page_title="Retail Sales Dashboard Premium", layout="wide")

# Titolo principale
st.title("📊 Retail Sales Analytics Dashboard Premium")

# Creazione delle tab principali
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Dashboard Principale", 
    "📈 Analisi Avanzate", 
    "🤖 Machine Learning",
    "🔌 API Integration",
    "⚙️ Settings"
])

# Inizializzazione delle session state
if 'data' not in st.session_state:
    st.session_state.data = None

if 'settings' not in st.session_state:
    st.session_state.settings = {
        'theme': 'Light',
        'currency': 'USD',
        'language': 'IT',
        'date_format': '%Y-%m-%d',
        'chart_style': 'Default',
        'auto_refresh': False,
        'refresh_interval': 5
    }

if 'api_settings' not in st.session_state:
    st.session_state.api_settings = {
        'api_key': '',
        'endpoint': '',
        'last_sync': None,
        'auto_sync': False
    }

# Tab 1: Dashboard Principale
with tab1:
    # Percorso del file predefinito
    default_file_path = "sample_data/retail_sales_dataset.csv"

    # Upload del file
    uploaded_file = st.file_uploader("Carica il file CSV delle vendite", type=['csv'])
    
    if uploaded_file is not None:
        # Caricamento e salvataggio dei dati nella session state
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("✅ File caricato correttamente!")
    else:
        try:
            # Caricamento del file predefinito
            st.session_state.data = pd.read_csv(default_file_path)
            st.info("ℹ️ Nessun file caricato. Mostrando i dati di esempio.")
        except FileNotFoundError:
            st.warning("⚠️ Nessun file caricato e file di esempio non trovato.")
            st.stop()

    # Se il dataset è stato caricato correttamente
    if st.session_state.data is not None:
        st.session_state.data['Date'] = pd.to_datetime(st.session_state.data['Date'])
        df = st.session_state.data

        # Sidebar per i filtri
        st.sidebar.header("📑 Filtri")

        # Filtro data
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        date_range = st.sidebar.date_input(
            "Seleziona intervallo date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # Filtri categoria e genere
        category_filter = st.sidebar.multiselect(
            "Categoria Prodotto",
            options=df['Product Category'].unique(),
            default=df['Product Category'].unique()
        )

        gender_filter = st.sidebar.multiselect(
            "Genere",
            options=df['Gender'].unique(),
            default=df['Gender'].unique()
        )

        # Applicazione filtri
        mask = (
            (df['Date'].dt.date >= date_range[0]) & 
            (df['Date'].dt.date <= date_range[1]) & 
            (df['Product Category'].isin(category_filter)) & 
            (df['Gender'].isin(gender_filter))
        )
        filtered_df = df[mask]

        # Layout principale
        col1, col2, col3, col4 = st.columns(4)

        # KPI Cards
        with col1:
            st.info(f"Vendite Totali\n${filtered_df['Total Amount'].sum():,.2f}")

        with col2:
            st.info(f"Numero Ordini\n{len(filtered_df):,}")

        with col3:
            st.info(f"Valore Medio Ordine\n${filtered_df['Total Amount'].mean():,.2f}")

        with col4:
            st.info(f"Clienti Unici\n{filtered_df['Customer ID'].nunique():,}")

        # Grafici
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Vendite per Categoria")
            fig_category = px.bar(
                filtered_df.groupby('Product Category')['Total Amount'].sum().reset_index(),
                x='Product Category',
                y='Total Amount',
                color='Product Category',
                title='Vendite Totali per Categoria'
            )
            st.plotly_chart(fig_category, use_container_width=True)

        with col_right:
            st.subheader("Distribuzione Vendite per Genere")
            fig_gender = px.pie(
                filtered_df,
                names='Gender',
                values='Total Amount',
                title='Distribuzione Vendite per Genere'
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        # Trend temporale
        st.subheader("Trend Vendite nel Tempo")
        daily_sales = filtered_df.groupby('Date')['Total Amount'].sum().reset_index()
        fig_trend = px.line(
            daily_sales,
            x='Date',
            y='Total Amount',
            title='Trend Vendite Giornaliere'
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Analisi età clienti
        st.subheader("Distribuzione Età Clienti")
        fig_age = px.histogram(
            filtered_df,
            x='Age',
            nbins=20,
            title='Distribuzione Età Clienti'
        )
        st.plotly_chart(fig_age, use_container_width=True)

        # Tabella dettagli
        st.subheader("Dettaglio Transazioni")
        st.dataframe(
            filtered_df.sort_values('Date', ascending=False),
            hide_index=True
        )

        # Statistiche aggiuntive
        st.subheader("Statistiche Aggiuntive")
        col_stats1, col_stats2 = st.columns(2)

        with col_stats1:
            st.write("Top 5 Giorni per Vendite")
            top_days = filtered_df.groupby('Date')['Total Amount'].sum().sort_values(ascending=False).head()
            st.write(top_days)

        with col_stats2:
            st.write("Statistiche per Categoria")
            category_stats = filtered_df.groupby('Product Category').agg({
                'Total Amount': ['sum', 'mean', 'count']
            }).round(2)
            st.write(category_stats)


# Tab 2: Analisi Avanzate
with tab2:
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.subheader("📊 Analisi Avanzate delle Vendite")
        
        # Analisi stagionale
        st.write("### 🗓️ Analisi Stagionale")
        daily_sales = df.groupby('Date')['Total Amount'].sum().reset_index()
        daily_sales.set_index('Date', inplace=True)
        
        # Decomposizione stagionale
        decomposition = seasonal_decompose(daily_sales['Total Amount'], period=30, extrapolate_trend='freq')
        
        col1, col2 = st.columns(2)
        with col1:
            fig_trend = px.line(decomposition.trend, title='Trend delle Vendite')
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            fig_seasonal = px.line(decomposition.seasonal, title='Pattern Stagionale')
            st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Analisi per fascia oraria e giorno della settimana
        st.write("### 📈 Analisi Dettagliate")
        
        # Analisi età e spesa
        age_spending = df.groupby(pd.qcut(df['Age'], q=5))['Total Amount'].agg(['mean', 'count', 'sum']).round(2)
        st.write("#### 👥 Analisi Spesa per Fasce d'Età")
        st.write(age_spending)
        
        # Matrice di correlazione
        st.write("#### 🔗 Correlazioni tra Variabili")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix,
                            title="Matrice di Correlazione",
                            color_continuous_scale='RdBu')
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("⚠️ Carica prima un file CSV nella Dashboard Principale")

# Tab 3: Machine Learning
with tab3:
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.subheader("🤖 Previsioni con Machine Learning")
        
        # Preparazione dati
        daily_sales = df.groupby('Date')['Total Amount'].sum().reset_index()
        daily_sales['Day_Number'] = (daily_sales['Date'] - daily_sales['Date'].min()).dt.days
        
        # Parametri del modello
        col1, col2 = st.columns(2)
        with col1:
            forecast_days = st.slider('Giorni di previsione', 7, 90, 30)
        with col2:
            train_size = st.slider('Percentuale dati per training', 50, 90, 80)
        
        # Training del modello
        train_size = int(len(daily_sales) * (train_size/100))
        train_data = daily_sales[:train_size]
        test_data = daily_sales[train_size:]
        
        model = LinearRegression()
        X_train = train_data[['Day_Number']]
        y_train = train_data['Total Amount']
        model.fit(X_train, y_train)
        
        # Previsioni
        future_days = pd.DataFrame({
            'Day_Number': range(daily_sales['Day_Number'].max() + 1,
                              daily_sales['Day_Number'].max() + forecast_days + 1)
        })
        
        future_predictions = model.predict(future_days)
        
        # Visualizzazione risultati
        fig_forecast = go.Figure()
        
        # Dati storici
        fig_forecast.add_trace(go.Scatter(
            x=daily_sales['Date'],
            y=daily_sales['Total Amount'],
            name='Dati Storici',
            line=dict(color='blue')
        ))
        
        # Previsioni
        future_dates = [daily_sales['Date'].max() + timedelta(days=x)
                       for x in range(1, forecast_days + 1)]
        fig_forecast.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            name='Previsione',
            line=dict(color='red', dash='dash')
        ))
        
        fig_forecast.update_layout(title='Previsione Vendite Future')
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Metriche del modello
        col1, col2, col3 = st.columns(3)
        with col1:
            r2 = r2_score(test_data['Total Amount'],
                         model.predict(test_data[['Day_Number']]))
            st.metric("R² Score", f"{r2:.3f}")
        
        with col2:
            mse = mean_squared_error(test_data['Total Amount'],
                                   model.predict(test_data[['Day_Number']]))
            st.metric("MSE", f"{mse:.2f}")
        
        with col3:
            predicted_total = future_predictions.sum()
            st.metric("Vendite Previste Totali", f"${predicted_total:,.2f}")
            
        # Spiegazione del modello
        with st.expander("ℹ️ Informazioni sul Modello"):
            st.write("""
            Questo modello utilizza la regressione lineare per prevedere le vendite future basandosi sui dati storici.
            - R² Score: indica quanto bene il modello si adatta ai dati (più vicino a 1 è meglio)
            - MSE: errore quadratico medio (più basso è meglio)
            - Le previsioni sono basate sul trend storico delle vendite
            """)
    else:
        st.warning("⚠️ Carica prima un file CSV nella Dashboard Principale")

# Tab 4: API Integration
with tab4:
    st.subheader("🔌 API Integration")
    
    # Sezione configurazione API
    st.write("### API Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        api_key = st.text_input(
            "API Key",
            value=st.session_state.api_settings['api_key'],
            type="password"
        )
        api_endpoint = st.text_input(
            "API Endpoint",
            value=st.session_state.api_settings['endpoint']
        )
        
    with col2:
        st.write("Last Sync:", st.session_state.api_settings['last_sync'] or "Never")
        auto_sync = st.toggle("Enable Auto Sync", st.session_state.api_settings['auto_sync'])
    
    # Pulsanti azione API
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Test Connection"):
            st.success("✅ Connection successful!")
    
    with col2:
        if st.button("Sync Data"):
            st.info("🔄 Synchronizing data...")
            
    with col3:
        if st.button("Export to API"):
            st.info("📤 Exporting data...")
            
    # Documentation
    with st.expander("📚 API Documentation"):
        st.write("""
        ### API Endpoints
        - GET /sales - Retrieve sales data
        - POST /sales - Upload sales data
        - GET /analytics - Retrieve analytics
        
        ### Authentication
        Add your API key in the header:
        ```
        Authorization: Bearer YOUR_API_KEY
        ```
        
        ### Rate Limits
        - 1000 requests per hour
        - 10000 requests per day
        """)

# Tab 5: Settings
with tab5:
    st.subheader("⚙️ Settings")
    
    # Interface Settings
    st.write("### Interface Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox(
            "Theme",
            ["Light", "Dark", "System"],
            index=["Light", "Dark", "System"].index(st.session_state.settings['theme'])
        )
        currency = st.selectbox(
            "Currency",
            ["USD", "EUR", "GBP", "JPY"],
            index=["USD", "EUR", "GBP", "JPY"].index(st.session_state.settings['currency'])
        )

    with col2:
        language = st.selectbox(
            "Language",
            ["IT", "EN", "FR", "DE"],
            index=["IT", "EN", "FR", "DE"].index(st.session_state.settings['language'])
        )
        date_format = st.text_input(
            "Date Format",
            value=st.session_state.settings['date_format']
        )

    # Chart Style
    st.write("### Chart Style")
    chart_style = st.selectbox(
        "Chart Style",
        ["Default", "Modern", "Classic"],
        index=["Default", "Modern", "Classic"].index(st.session_state.settings['chart_style'])
    )

    # Auto-Refresh Settings
    st.write("### Auto-Refresh Settings")
    col1, col2 = st.columns(2)
    with col1:
        auto_refresh = st.checkbox(
            "Enable Auto-Refresh",
            value=st.session_state.settings['auto_refresh']
        )
    with col2:
        refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            1, 60, st.session_state.settings['refresh_interval']
        )

    # Save Settings Button
    if st.button("💾 Save Settings"):
        # Aggiorna lo stato della sessione
        st.session_state.settings.update({
            'theme': theme,
            'currency': currency,
            'language': language,
            'date_format': date_format,
            'chart_style': chart_style,
            'auto_refresh': auto_refresh,
            'refresh_interval': refresh_interval
        })
        
        # Cambia il tema dinamicamente
        if theme == "Light":
            st.markdown(
                """
                <style>
                .stApp {
                    background-color: #FFFFFF;
                    color: #000000;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        elif theme == "Dark":
            st.markdown(
                """
                <style>
                .stApp {
                    background-color: #262730;
                    color: #FFFFFF;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

        # Notifica all'utente
        st.success("✅ Settings updated successfully!")
        
        # Ricarica la pagina
        st.experimental_rerun()
