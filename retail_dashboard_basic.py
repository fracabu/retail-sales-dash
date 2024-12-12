import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configurazione della pagina
st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

# Titolo principale
st.title("📊 Retail Sales Analytics Dashboard")

# Upload del file
uploaded_file = st.file_uploader("Carica il file CSV delle vendite", type=['csv'])

if uploaded_file is not None:
    # Caricamento dati
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

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
        column_config={
            "Date": st.column_config.DateColumn("Data"),
            "Total Amount": st.column_config.NumberColumn(
                "Importo Totale",
                format="$%.2f"
            )
        },
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

else:
    st.write("⬅️ Carica un file CSV per visualizzare il dashboard")