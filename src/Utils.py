import numpy as np
from scipy.signal import welch

def estrai_de_versatile(data, fs=1000, batch_size=500, 
                        bands_to_extract=['Alpha', 'Beta', 'Gamma'], 
                        average_bands=False):
    """
    Estrae l'Entropia Differenziale (DE) in modo parametrico.
    
    Input: data shape (N, 62, 1500)
    Output (average_bands=True):  (N, 62, numero_di_bande_selezionate)
    Output (average_bands=False): (N, 62, somma_dei_bin_delle_bande_selezionate)
    """
    N = data.shape[0]
    num_channels = data.shape[1]
    time_steps = data.shape[2]
    
    # 1. Definizione rigorosa dei range standard
    BANDS_DEF = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 14),
        'Beta':  (14, 31),
        'Gamma': (31, 50)
    }
    
    # Validazione dell'input
    for b in bands_to_extract:
        if b not in BANDS_DEF:
            raise ValueError(f"Errore: Banda '{b}' non riconosciuta. Usa: {list(BANDS_DEF.keys())}")
            
    # 2. Pre-calcolo delle frequenze
    freqs, _ = welch(np.zeros(time_steps), fs=fs, nperseg=time_steps)
    
    # 3. Allocazione condizionale in base alla logica scelta
    if average_bands:
        num_features = len(bands_to_extract)
        de_features = np.zeros((N, num_channels, num_features), dtype='float32')
        # Creiamo una lista di maschere separate per ogni banda
        masks = [np.logical_and(freqs >= BANDS_DEF[b][0], freqs <= BANDS_DEF[b][1]) for b in bands_to_extract]
    else:
        # Creiamo un'unica maschera booleana combinata per tutte le bande richieste
        combined_mask = np.zeros_like(freqs, dtype=bool)
        for b in bands_to_extract:
            combined_mask |= np.logical_and(freqs >= BANDS_DEF[b][0], freqs <= BANDS_DEF[b][1])
        
        num_features = np.sum(combined_mask)
        de_features = np.zeros((N, num_channels, num_features), dtype='float32')

    print(f"Estrazione DE | Media={average_bands} | Bande={bands_to_extract}")
    print(f"Campioni: {N} | Feature estratte per canale: {num_features}")
    
    # 4. Elaborazione a blocchi (Chunking)
    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        batch_data = data[i:end_idx]
        
        # Welch vettorizzato sul batch
        _, psd_batch = welch(batch_data, fs=fs, nperseg=time_steps, axis=-1)
        
        if average_bands:
            # Calcolo isolato e mediato per singola banda
            for b_idx, mask in enumerate(masks):
                mean_pwrs = np.mean(psd_batch[:, :, mask], axis=-1)
                mean_pwrs = np.maximum(mean_pwrs, 1e-12) # Sicurezza matematica
                de_features[i:end_idx, :, b_idx] = 0.5 * np.log(2 * np.pi * np.e * mean_pwrs)
        else:
            # Calcolo granulare sui bin contigui
            pwrs_batch = psd_batch[:, :, combined_mask]
            pwrs_batch = np.maximum(pwrs_batch, 1e-12)
            de_features[i:end_idx, :, :] = 0.5 * np.log(2 * np.pi * np.e * pwrs_batch)
            
        if end_idx % (batch_size * 2) == 0 or end_idx == N:
            print(f"  -> Processati: {end_idx}/{N}")
            
    return de_features