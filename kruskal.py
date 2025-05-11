import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import matplotlib.patches as mpatches
import time
import io
from PIL import Image

# Configurazione pagina Streamlit
st.set_page_config(layout="wide", page_title="MDS con algoritmo di Kruskal", page_icon="📊")

# Titolo dell'applicazione
st.title("Analisi MDS delle preferenze sulle serie TV Sky Italia")

# Descrizione dell'analisi nella sidebar
with st.sidebar:
    st.header("Informazioni")
    st.info("""
    Questa applicazione analizza le preferenze di 10 spettatori su 8 serie TV Sky Italia, 
    visualizzando le similarità tra serie usando l'algoritmo MDS (Multidimensional Scaling) 
    con il metodo di Kruskal.
    
    **Scala di valutazione:**  
    - 1: Non mi piace
    - 5: Mi piace tantissimo
    
    Puoi visualizzare le iterazioni dell'algoritmo e il confronto tra le distanze originali e quelle MDS.
    """)
    
    # Parametri dell'algoritmo MDS
    st.header("Parametri MDS")
    n_dimensions = st.radio("Numero di dimensioni:", [2, 3], horizontal=True)
    n_iterations = st.slider("Numero di iterazioni:", min_value=2, max_value=10, value=3)
    random_seed = st.number_input("Seed casuale:", min_value=0, max_value=1000, value=42)

# Inizializzazione dei dati
@st.cache_data
def load_data():
    # Creazione della tabella di preferenze
    data = {
        'Spettatore': list(range(1, 11)),
        'Sex and the City': [5, 3, 4, 5, 5, 5, 3, 5, 4, 3],
        'Il Trono di Spade': [4, 5, 5, 4, 4, 3, 5, 4, 5, 4],
        'Chernobyl': [3, 5, 5, 2, 5, 4, 5, 2, 3, 5],
        'Succession': [2, 4, 3, 3, 4, 2, 4, 3, 4, 4],
        'The Last of Us': [5, 4, 5, 4, 4, 4, 5, 3, 5, 4],
        'The Young Pope': [1, 3, 2, 1, 3, 3, 3, 3, 2, 3],
        'True Detective': [4, 4, 3, 3, 2, 4, 2, 5, 5, 3],
        'The Walking Dead': [2, 1, 1, 2, 1, 1, 1, 1, 1, 2]
    }
    return pd.DataFrame(data)

df = load_data()

#################################
# 1. TABELLA DELLE PREFERENZE
#################################
st.header("1. Tabella delle preferenze")
st.write("""
La seguente tabella mostra le valutazioni di 10 spettatori per 8 delle serie TV Sky Italia più famose. 
Le celle con le valutazioni più alte sono evidenziate in verde, mentre quelle con le valutazioni più basse sono evidenziate in rosso.
""")

# Mostra la tabella di preferenze con evidenziate le celle di valori massimi e minimi
st.dataframe(df.set_index('Spettatore').style.highlight_max(axis=1, color='#90EE90').highlight_min(axis=1, color='#FFCCCC'))

# Estrazione delle colonne relative alle serie TV
serie_tv = df.columns[1:].tolist()
ratings = df[serie_tv].values.T  # Trasposta per avere serie TV sulle righe

#################################
# 2. ANALISI COMPARATIVA A COPPIE
#################################
st.header("2. Tabella di ordinamento a coppie")
st.write("""
La tabella seguente rappresenta la comparazione diretta tra coppie di serie TV, 
mostrando quante volte una serie (riga) è stata preferita rispetto ad un'altra (colonna) 
da parte degli spettatori.
""")

# Creiamo una matrice di comparazione a coppie
def create_pairwise_comparison(df, serie_tv):
    n_series = len(serie_tv)
    comparison_matrix = np.zeros((n_series, n_series))
    
    # Per ogni spettatore, confronta le serie a coppie
    for _, row in df.iterrows():
        for i, serie1 in enumerate(serie_tv):
            for j, serie2 in enumerate(serie_tv):
                if i != j:  # Non confrontare una serie con se stessa
                    if row[serie1] > row[serie2]:
                        comparison_matrix[i, j] += 1
    
    return pd.DataFrame(comparison_matrix, index=serie_tv, columns=serie_tv)

# Calcola la matrice di comparazione a coppie
pairwise_df = create_pairwise_comparison(df, serie_tv)

# Visualizza la matrice con heatmap
st.dataframe(pairwise_df.style.background_gradient(cmap='Blues', axis=None))

# Analisi della dominanza
dominance_scores = pairwise_df.sum(axis=1).sort_values(ascending=False)
st.subheader("Analisi della dominanza (vittorie)")
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(dominance_scores.index, dominance_scores.values, color=plt.cm.viridis(np.linspace(0, 1, len(dominance_scores))))
    
    # Aggiungi etichette con i valori
    for i, v in enumerate(dominance_scores.values):
        ax.text(i, v + 0.5, f"{v:.0f}", ha='center')
    
    ax.set_title("Numero di 'vittorie' nei confronti a coppie")
    ax.set_xlabel("Serie TV")
    ax.set_ylabel("Numero di vittorie")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.write("""
    **Risultati principali:**
    
    - **Il Trono di Spade** e **The Last of Us** sono le serie più preferite, vincendo la maggior parte dei confronti con altre serie.
    
    - **The Walking Dead** è chiaramente la meno preferita, non vincendo contro nessun'altra serie.
    
    - **The Young Pope** ha una base di fan più limitata, vincendo principalmente contro The Walking Dead.
    """)

#################################
# 3. GENERAZIONE COORDINATE CASUALI E MDS
#################################
st.header("3. Algoritmo MDS con Kruskal")

# Calcolo della matrice di dissimilarità (distanze euclidee)
distances = pdist(ratings, metric='euclidean')
dist_matrix = squareform(distances)

# Funzione per calcolare il valore di stress
def calculate_stress(dist_matrix, positions):
    """Calcola il valore di stress (formula di Kruskal)"""
    n = positions.shape[0]
    pos_dist = np.zeros((n, n))
    
    # Calcola la matrice delle distanze dalle posizioni
    for i in range(n):
        for j in range(i+1, n):
            pos_dist[i, j] = np.sqrt(np.sum((positions[i] - positions[j])**2))
            pos_dist[j, i] = pos_dist[i, j]
    
    # Calcola lo stress (formula di Kruskal)
    numerator = np.sum((dist_matrix - pos_dist)**2)
    denominator = np.sum(dist_matrix**2)
    
    return np.sqrt(numerator / denominator), pos_dist

# Genera posizioni iniziali casuali
np.random.seed(random_seed)
initial_pos = np.random.rand(len(serie_tv), n_dimensions) * 2 - 1  # Valori tra -1 e 1

st.subheader("3.1 Posizioni iniziali casuali")
st.write(f"Coordinate iniziali generate con seed: {random_seed}")

# Mostra le coordinate iniziali
initial_coords_df = pd.DataFrame(
    initial_pos, 
    index=serie_tv,
    columns=[f"Dimensione {i+1}" for i in range(n_dimensions)]
)
st.dataframe(initial_coords_df)

# Visualizza le posizioni iniziali
if n_dimensions == 2:
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(serie_tv)))
    
    for i, serie in enumerate(serie_tv):
        ax.scatter(initial_pos[i, 0], initial_pos[i, 1], color=colors[i], s=100)
        ax.text(initial_pos[i, 0], initial_pos[i, 1], serie, fontsize=10, ha='center', va='bottom')
    
    ax.set_title('Posizioni iniziali casuali (2D)')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    st.pyplot(fig)
else:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, len(serie_tv)))
    
    for i, serie in enumerate(serie_tv):
        ax.scatter(initial_pos[i, 0], initial_pos[i, 1], initial_pos[i, 2], color=colors[i], s=100)
        ax.text(initial_pos[i, 0], initial_pos[i, 1], initial_pos[i, 2], serie, fontsize=9)
    
    ax.set_title('Posizioni iniziali casuali (3D)')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
    st.pyplot(fig)

st.subheader("3.2 Matrice delle distanze originale")
st.write("Matrice di dissimilarità tra le serie TV basata sui voti degli spettatori (calcolata con distanza euclidea):")

# Creazione di un DataFrame per visualizzare la matrice delle distanze
dist_df = pd.DataFrame(dist_matrix, index=serie_tv, columns=serie_tv)
st.dataframe(dist_df.style.background_gradient(cmap='viridis', axis=None))

st.subheader("3.3 Iterazioni dell'algoritmo di Kruskal")
st.write(f"Esecuzione di {n_iterations} iterazioni dell'algoritmo MDS con il metodo di Kruskal:")

# Funzione per implementare l'algoritmo MDS con iterazioni visibili
def kruskal_mds(dist_matrix, n_components=2, max_iter=3, random_state=None):
    """Implementa l'algoritmo MDS per mostrare le iterazioni"""
    n = dist_matrix.shape[0]
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Inizializza le posizioni casualmente
    X = np.random.rand(n, n_components) * 2 - 1  # Valori tra -1 e 1
    
    # Lista per memorizzare le posizioni, stress e matrici di distanza a ogni iterazione
    all_positions = [X.copy()]
    all_stress = []
    all_distance_matrices = []
    
    # Calcola stress iniziale
    initial_stress, initial_dist_matrix = calculate_stress(dist_matrix, X)
    all_stress.append(initial_stress)
    all_distance_matrices.append(initial_dist_matrix)
    
    # Esegui le iterazioni
    for i in range(max_iter):
        # Calcola la matrice delle distanze dalla configurazione attuale
        current_dist = np.zeros((n, n))
        for a in range(n):
            for b in range(a+1, n):
                current_dist[a, b] = np.sqrt(np.sum((X[a] - X[b])**2))
                current_dist[b, a] = current_dist[a, b]
        
        # Calcola il gradiente per ogni punto
        grad = np.zeros((n, n_components))
        for a in range(n):
            for b in range(n):
                if a != b and current_dist[a, b] > 1e-10:  # Evita divisione per zero
                    # Differenza tra distanza desiderata e distanza attuale
                    diff = dist_matrix[a, b] - current_dist[a, b]
                    # Aggiorna il gradiente
                    grad[a] += diff * (X[a] - X[b]) / current_dist[a, b]
        
        # Aggiorna le posizioni (passo di gradient descent)
        step_size = 0.1  # Iperparametro da regolare
        X = X - step_size * grad
        
        # Calcola stress e matrice di distanze per questa iterazione
        iter_stress, iter_dist_matrix = calculate_stress(dist_matrix, X)
        
        # Salva la configurazione corrente
        all_positions.append(X.copy())
        all_stress.append(iter_stress)
        all_distance_matrices.append(iter_dist_matrix)
    
    return all_positions, all_stress, all_distance_matrices

# Esegui MDS personalizzato
positions, stress_values, dist_matrices = kruskal_mds(
    dist_matrix, n_components=n_dimensions, max_iter=n_iterations, random_state=random_seed
)

# Visualizzazione dell'evoluzione dello stress
st.write("**Evoluzione del valore di stress:**")
fig_stress, ax_stress = plt.subplots(figsize=(10, 4))
ax_stress.plot(range(n_iterations + 1), stress_values, 'o-', color='blue')
ax_stress.set_xlabel('Iterazione')
ax_stress.set_ylabel('Stress')
ax_stress.set_title('Convergenza dell\'algoritmo MDS')
ax_stress.grid(True, linestyle='--', alpha=0.7)

# Aggiungi linee orizzontali tratteggiate e valori
for i, s in enumerate(stress_values):
    ax_stress.axhline(y=s, color='gray', linestyle=':', alpha=0.5)
    ax_stress.text(i+0.1, s, f"{s:.4f}", va='center', fontsize=8)

st.pyplot(fig_stress)

# Visualizza ogni iterazione
st.write("**Visualizzazione delle configurazioni MDS per ogni iterazione:**")
tabs = st.tabs([f"Iterazione {i}" for i in range(n_iterations + 1)])

for i, (pos, tab) in enumerate(zip(positions, tabs)):
    with tab:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Visualizza le coordinate
            pos_df = pd.DataFrame(
                pos, 
                index=serie_tv,
                columns=[f"Dimensione {j+1}" for j in range(n_dimensions)]
            )
            st.write(f"**Coordinate MDS - Iterazione {i}:**")
            st.dataframe(pos_df)
            
            # Visualizza la configurazione MDS
            if n_dimensions == 2:
                fig, ax = plt.subplots(figsize=(8, 8))
                colors = plt.cm.tab10(np.linspace(0, 1, len(serie_tv)))
                
                for j, serie in enumerate(serie_tv):
                    ax.scatter(pos[j, 0], pos[j, 1], color=colors[j], s=100)
                    ax.text(pos[j, 0], pos[j, 1], serie, fontsize=10, ha='center', va='bottom')
                
                ax.set_title(f'MDS 2D - Iterazione {i}')
                ax.set_xlim([-2, 2])
                ax.set_ylim([-2, 2])
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                plt.figtext(0.02, 0.02, f"Stress: {stress_values[i]:.4f}", fontsize=8)
                
                st.pyplot(fig)
            else:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                colors = plt.cm.tab10(np.linspace(0, 1, len(serie_tv)))
                
                for j, serie in enumerate(serie_tv):
                    ax.scatter(pos[j, 0], pos[j, 1], pos[j, 2], color=colors[j], s=80)
                    ax.text(pos[j, 0], pos[j, 1], pos[j, 2], serie, fontsize=8)
                
                ax.set_title(f'MDS 3D - Iterazione {i}')
                ax.set_xlim([-2, 2])
                ax.set_ylim([-2, 2])
                ax.set_zlim([-2, 2])
                plt.figtext(0.02, 0.02, f"Stress: {stress_values[i]:.4f}", fontsize=8)
                
                st.pyplot(fig)
        
        with col2:
            # Matrice delle distanze per questa iterazione
            st.write(f"**Matrice delle distanze - Iterazione {i}:**")
            iter_dist_df = pd.DataFrame(dist_matrices[i], index=serie_tv, columns=serie_tv)
            st.dataframe(iter_dist_df.style.background_gradient(cmap='viridis', axis=None))
            
st.subheader("3.4 Confronto tra distanze originali e distanze MDS")

# Confronto ordinato tra le distanze originali e quelle MDS
dist_orig_flat = []
dist_mds_flat = []
labels = []

for i in range(len(serie_tv)):
    for j in range(i+1, len(serie_tv)):
        dist_orig_flat.append(dist_matrix[i, j])
        dist_mds_flat.append(dist_matrices[-1][i, j])
        labels.append(f"{serie_tv[i]}-{serie_tv[j]}")

# Crea un DataFrame per la visualizzazione e ordina per distanza originale
comparison_df = pd.DataFrame({
    'Coppia': labels,
    'Distanza originale': dist_orig_flat,
    'Distanza MDS': dist_mds_flat,
    'Differenza': np.array(dist_orig_flat) - np.array(dist_mds_flat)
})

# Ordina per distanza originale
comparison_df_sorted = comparison_df.sort_values('Distanza originale')

# Visualizza il DataFrame ordinato
st.write("**Confronto ordinato delle distanze:**")
st.dataframe(comparison_df_sorted.style.background_gradient(subset=['Differenza'], cmap='coolwarm'))

# Crea un grafico di confronto
st.write("**Grafico di confronto delle distanze:**")
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot dei punti
ax.scatter(dist_orig_flat, dist_mds_flat, alpha=0.7)
ax.plot([min(dist_orig_flat), max(dist_orig_flat)], [min(dist_orig_flat), max(dist_orig_flat)], 'r--', alpha=0.5, label="Corrispondenza perfetta")

# Linea di regressione
z = np.polyfit(dist_orig_flat, dist_mds_flat, 1)
p = np.poly1d(z)
ax.plot(sorted(dist_orig_flat), p(sorted(dist_orig_flat)), 'g-', alpha=0.7, label=f"Regressione (y = {z[0]:.2f}x + {z[1]:.2f})")

# Evidenzia i punti con maggiore discrepanza
top_diff_idx = np.argsort(np.abs(np.array(dist_orig_flat) - np.array(dist_mds_flat)))[-5:]
for idx in top_diff_idx:
    ax.annotate(labels[idx], 
                (dist_orig_flat[idx], dist_mds_flat[idx]),
                fontsize=9, alpha=0.8)

ax.set_xlabel('Distanze originali')
ax.set_ylabel('Distanze MDS')
ax.set_title('Confronto tra distanze originali e distanze MDS')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()

st.pyplot(fig)

# Correlazione di Pearson tra le distanze originali e quelle MDS
corr = np.corrcoef(dist_orig_flat, dist_mds_flat)[0, 1]
st.write(f"**Correlazione tra distanze originali e distanze MDS:** {corr:.4f}")

st.subheader("3.5 Elementi non corrispondenti e aggiornamento della matrice")

# Identifica gli elementi con maggiore discrepanza
threshold = 0.5 * np.std(comparison_df['Differenza'])
non_matching = comparison_df[np.abs(comparison_df['Differenza']) > threshold]

st.write(f"**Elementi con discrepanza significativa (soglia: {threshold:.4f}):**")
st.dataframe(non_matching.sort_values('Differenza', key=abs, ascending=False))

# Crea una nuova matrice sostituendo i valori non corrispondenti con la media
new_matrix = dist_matrix.copy()
for idx, row in non_matching.iterrows():
    coppia = row['Coppia'].split('-')
    i = serie_tv.index(coppia[0])
    j = serie_tv.index(coppia[1])
    
    # Calcola la media tra distanza originale e distanza MDS
    new_value = (dist_matrix[i, j] + dist_matrices[-1][i, j]) / 2
    
    # Aggiorna la matrice
    new_matrix[i, j] = new_value
    new_matrix[j, i] = new_value

# Visualizza la nuova matrice
st.write("**Nuova matrice di distanze (con valori aggiornati):**")
new_dist_df = pd.DataFrame(new_matrix, index=serie_tv, columns=serie_tv)
st.dataframe(new_dist_df.style.background_gradient(cmap='viridis', axis=None))

# Esegui MDS con la nuova matrice
st.subheader("3.6 MDS con la matrice aggiornata")

mds = MDS(n_components=n_dimensions, dissimilarity='precomputed', random_state=random_seed)
new_pos = mds.fit_transform(new_matrix)

# Visualizza le nuove coordinate
new_pos_df = pd.DataFrame(
    new_pos, 
    index=serie_tv,
    columns=[f"Dimensione {i+1}" for i in range(n_dimensions)]
)

st.write("**Nuove coordinate MDS:**")
st.dataframe(new_pos_df)

# Visualizza la nuova configurazione MDS
if n_dimensions == 2:
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(serie_tv)))
    
    for i, serie in enumerate(serie_tv):
        ax.scatter(new_pos[i, 0], new_pos[i, 1], color=colors[i], s=100)
        ax.text(new_pos[i, 0], new_pos[i, 1], serie, fontsize=12, ha='center', va='bottom')
    
    ax.set_title('MDS con matrice aggiornata')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    st.pyplot(fig)
else:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, len(serie_tv)))
    
    for i, serie in enumerate(serie_tv):
        ax.scatter(new_pos[i, 0], new_pos[i, 1], new_pos[i, 2], color=colors[i], s=100)
        ax.text(new_pos[i, 0], new_pos[i, 1], new_pos[i, 2], serie, fontsize=9)
    
    ax.set_title('MDS 3D con matrice aggiornata')
    
    st.pyplot(fig)

# Calcola lo stress della nuova configurazione
new_stress, _ = calculate_stress(dist_matrix, new_pos)
st.write(f"**Stress della nuova configurazione:** {new_stress:.4f}")
st.write(f"**Stress della configurazione originale (ultima iterazione):** {stress_values[-1]:.4f}")

# Confronto tra la configurazione finale originale e quella nuova
st.subheader("3.7 Confronto tra configurazioni MDS")

# Visualizza entrambe le configurazioni su un unico grafico
if n_dimensions == 2:
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(serie_tv)))
    
    # Configurazione finale originale (punti)
    for i, serie in enumerate(serie_tv):
        ax.scatter(positions[-1][i, 0], positions[-1][i, 1], color=colors[i], s=100, alpha=0.7, label=None)
        ax.text(positions[-1][i, 0], positions[-1][i, 1], serie, fontsize=10, ha='center', va='bottom')
    
    # Nuova configurazione (cerchi)
    for i, serie in enumerate(serie_tv):
        ax.scatter(new_pos[i, 0], new_pos[i, 1], edgecolor=colors[i], facecolor='none', s=150, linewidth=2, label=serie)
        ax.text(new_pos[i, 0], new_pos[i, 1], serie, fontsize=10, ha='center', va='top')
    
    # Connetti le posizioni con linee
    for i, serie in enumerate(serie_tv):
        ax.plot([positions[-1][i, 0], new_pos[i, 0]], [positions[-1][i, 1], new_pos[i, 1]], '--', color=colors[i], alpha=0.5)
    
    ax.set_title('Confronto tra configurazioni MDS')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Legenda
    ax.plot([], [], 'o', color='gray', label='Configurazione originale')
    ax.plot([], [], 'o', mfc='none', mec='gray', label='Configurazione aggiornata')
    ax.legend(loc='best')
    
    st.pyplot(fig)
else:
    st.write("Il confronto grafico è disponibile solo per la visualizzazione 2D.")
with st.expander("Spiegazione dell'algoritmo MDS e del metodo di Kruskal"):
    st.write("""
    ### Multidimensional Scaling (MDS)
    Il Multidimensional Scaling (MDS) è una tecnica di analisi dei dati utilizzata per rappresentare in uno spazio a dimensione ridotta 
    (solitamente 2D o 3D) un insieme di oggetti, preservando il più possibile le distanze tra di essi. 
    È particolarmente utile per visualizzare relazioni in insiemi di dati complessi come matrici di dissimilarità o distanze.

    #### Processo:
    1. Si parte da una matrice di dissimilarità o di distanze tra coppie di oggetti.
    2. L'MDS cerca di trovare una configurazione di punti in uno spazio a dimensione ridotta (ad esempio 2D o 3D) 
       in cui le distanze tra i punti corrispondano il più possibile alle dissimilarità originali.
    3. Utilizza metodi come il calcolo degli autovalori per ridurre la dimensionalità, minimizzando una funzione di stress.

    ### Metodo di Kruskal
    Il metodo di Kruskal è un approccio classico per il MDS che minimizza una funzione chiamata "Stress".
    L'obiettivo è ottimizzare la rappresentazione in modo che le distanze nello spazio ridotto siano monotonicamente 
    correlate con le dissimilarità originali.

    #### Passaggi:
    1. Viene calcolata una matrice di distanze nello spazio ridotto.
    2. Viene calcolata una correlazione monotona tra le distanze ridotte e le dissimilarità originali.
    3. Il metodo cerca di ridurre lo "Stress", una misura della discrepanza tra le due.
    
    #### Formula dello Stress:
    \[
    Stress = \sqrt{\frac{\sum_{i < j} (d_{ij} - \delta_{ij})^2}{\sum_{i < j} \delta_{ij}^2}}
    \]
    dove \(d_{ij}\) rappresenta le distanze nello spazio ridotto e \(\delta_{ij}\) le dissimilarità originali.
    """)
