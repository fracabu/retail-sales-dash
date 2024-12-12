# 📊 Retail Sales Analytics Dashboard Premium

Una dashboard interattiva per analizzare i dati di vendita al dettaglio, esplorare approfondimenti avanzati e prevedere tendenze future utilizzando tecniche di Machine Learning. Progettata per essere facile da usare, consente di caricare dataset CSV personalizzati o di lavorare con un dataset di esempio predefinito.

---

## 🛠️ Funzionalità

### 1. **Dashboard Principale**
   - **Caricamento Dati**:
     - Carica un file CSV personalizzato con dati di vendita.
     - Visualizza automaticamente un dataset di esempio se non viene caricato alcun file.
   - **Filtri Avanzati**:
     - Intervallo di date.
     - Categorie di prodotto.
     - Genere del cliente.
   - **Indicatori KPI**:
     - Vendite totali.
     - Numero di ordini.
     - Valore medio ordine.
     - Clienti unici.
   - **Grafici Interattivi**:
     - Vendite per categoria.
     - Distribuzione delle vendite per genere.
     - Trend delle vendite giornaliere.
   - **Analisi Demografica**: Distribuzione per fasce d’età.
   - **Tabella Transazioni**: Visualizza i dettagli delle transazioni filtrate.

### 2. **Analisi Avanzate**
   - **Decomposizione Stagionale**: Analisi dei trend e pattern stagionali.
   - **Matrice di Correlazione**: Individua relazioni tra variabili numeriche.
   - **Analisi per Fasce d'Età**: Spesa media, numero di transazioni e vendite totali.

### 3. **Machine Learning**
   - **Modello di Previsione**: Prevedi le vendite future con la regressione lineare.
   - **Metriche del Modello**:
     - R² (coefficiente di determinazione).
     - Errore quadratico medio (MSE).
   - **Visualizzazione delle Previsioni**:
     - Dati storici.
     - Previsioni future.

### 4. **Integrazione API**
   - Configura chiavi API ed endpoint per sincronizzare o esportare dati.
   - Documentazione API integrata con esempi di utilizzo.

### 5. **Impostazioni**
   - Personalizza l'interfaccia utente:
     - Tema (Light/Dark/System).
     - Valuta.
     - Lingua.
     - Formato data.
     - Stile grafici.
   - Configura intervalli di aggiornamento automatico.

---

## 🚀 Installazione

### Prerequisiti
- Python 3.8 o superiore.
- Ambiente virtuale (consigliato).

### 1. Clona il Repository
```bash
git clone https://github.com/tuo-username/retail-sales-dashboard.git
cd retail-sales-dashboard
```

### 2. Crea e Attiva un Ambiente Virtuale
```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```

### 3. Installa le Dipendenze
```bash
pip install -r requirements.txt
```

### 4. Avvia l'Applicazione
```bash
streamlit run retail_dashboard_premium.py
```

---

## 📁 Struttura del Progetto

```plaintext
retail-sales-dashboard/
│
├── retail_dashboard_basic.py       # Versione base della dashboard
├── retail_dashboard_standard.py    # Versione standard della dashboard
├── retail_dashboard_premium.py     # Versione premium con tutte le funzionalità
├── requirements.txt                # Dipendenze del progetto
├── .gitignore                      # File ignorati da Git
├── README.md                       # Documentazione
├── sample_data/                    # Dataset di esempio
│   └── retail_sales_dataset.csv    # File CSV di esempio
├── venv/                           # Ambiente virtuale (escluso da Git)
```

---

## 📝 Dataset di Esempio

Il file `sample_data/retail_sales_dataset.csv` è incluso come dataset di esempio predefinito. Verrà caricato automaticamente se non viene fornito alcun file personalizzato.

#### **Formato delle Colonne**
- **Transaction ID**: Identificativo unico della transazione.
- **Date**: Data della transazione (Formato: `YYYY-MM-DD`).
- **Customer ID**: Identificativo unico del cliente.
- **Gender**: Genere del cliente (`Male`/`Female`).
- **Age**: Età del cliente.
- **Product Category**: Categoria del prodotto acquistato.
- **Quantity**: Quantità acquistata.
- **Price per Unit**: Prezzo unitario del prodotto.
- **Total Amount**: Totale della transazione.

#### **Esempio di Riga**
```csv
Transaction ID,Date,Customer ID,Gender,Age,Product Category,Quantity,Price per Unit,Total Amount
1,2024-01-01,C001,Male,34,Electronics,1,299.99,299.99
```

Puoi caricare il file nella dashboard per esplorare tutte le funzionalità.

---

## 📊 Esempi di Output

### Dashboard Principale
![Dashboard Principale](![alt text](image.png))

### Previsioni di Vendita
![Previsioni](![alt text](image-1.png))

---

## 🔧 Configurazione Avanzata

### File `.env` (Opzionale)
Puoi configurare variabili sensibili come chiavi API nel file `.env`:
```env
API_KEY=your_api_key_here
API_ENDPOINT=https://api.example.com
```

Assicurati che il file `.env` sia escluso da Git aggiungendolo al `.gitignore`.

---

## 🛠️ Supporto

In caso di problemi:
1. Controlla il file `README.md`.
2. Apri un [Issue](https://github.com/fracabu/retail-sales-dashboard/issues).

---

## 🏆 Contributi

I contributi sono benvenuti! Sentiti libero di aprire una pull request o suggerire miglioramenti.

---

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Consulta il file [LICENSE](LICENSE) per ulteriori dettagli.
```
