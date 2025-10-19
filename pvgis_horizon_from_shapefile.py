#!/usr/bin/env python
"""
Script: pvgis_horizon_from_shapefile.py

Descrizione:
- Estrae shapefile da un .zip
- Carica il layer degli edifici
- Seleziona l'edificio specificato dall'indice in configurazione come edificio under-study
- Calcola l'userhorizon ogni 10° (configurabile)
- Calcola l'orientamento del pannello (azimuth) con una regola deterministica
- Calcola la Potenza di Picco (peakpower) in base all'area dell'edificio
- Genera un plot migliorato (output_plot.png)
- Chiama l'API PVGIS (v5_2) 'seriescalc' con userhorizon, aspect e peakpower calcolato
- [FIX: Utilizza startyear/endyear per generare un solo anno di dati (8760 righe)]
- Calcola l'energia totale annua (kWh) dalla serie oraria
- Salva output_summary.json e output_hourly_data.csv

Uso:
    python pvgis_horizon_from_shapefile.py path/to/buildings.zip
    python pvgis_horizon_from_shapefile.py --run-orientation-tests

Output:
- stampa userhorizon (stringa pronta per PVGIS)
- stampa ORIENTATION SUMMARY
- stampa PEAKPOWER SUMMARY
- salva un PNG con la mappa e annotazioni (output_plot.png)
- salva un JSON con i risultati (output_summary.json)
- salva un CSV con i dati orari (output_hourly_data.csv)
- stampa il risultato JSON minimo ritornato da PVGIS

Note / Assunzioni:
- Il layer degli edifici deve contenere una proprietà 'height' (in metri) oppure 'levels' (numero di piani).
  Se nessuna proprietà è disponibile lo script usa un'altezza di default (DEFAULT_HEIGHT_M).
- Lo shapefile può essere in qualunque CRS: lo script riporta i dati in un CRS metrico (UTM) per i calcoli di distanza.
- Metodo scelto (horizon): geometrico 2D con trigonometria elementare (nessun DSM).
- Metodo (orientation): MBRect_rect_south_midpoint_preference

Requisiti Python (pip):
    geopandas
    shapely
    numpy
    matplotlib
    requests
    pandas
"""

import sys
import os
import zipfile
import tempfile
import math
import json
from pathlib import Path
import shutil
import pandas as pd
# Import originali
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import requests
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points

# ----------------------
# Configurazioni
# ----------------------
STEP_DEG = 10            # passo angolare per l'horizon (10° come richiesto)
RAY_LENGTH_M = 200      # lunghezza del raggio in metri
DEFAULT_HEIGHT_M = 10.0  # altezza di fallback (m)
ASSUME_LEVEL_HEIGHT = 3.0 # se si usa 'levels' -> metri per piano
PVGIS_ENDPOINT = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc" 
PLOT_OUTPUT = "output_plot.png"
SUMMARY_OUTPUT = "output_summary.json"
HOURLY_OUTPUT = "output_hourly_data.csv"
DEFAULT_TILT_FOR_ASPECT = 35 # Tilt (gradi) da usare quando si fornisce un aspect
ORIENTATION_UNCERTAINTY_RATIO = 1.05 # (long/short) ratio threshold for near-square
# Parametri per la stima di peakpower
ROOF_AREA_FACTOR = 0.4 # Fattore di utilizzo dell'area del tetto (40%)
POWER_DENSITY_W_PER_M2 = 200 # Densità di potenza stimata dei pannelli (W/m²)

# 🎯 INDICE DELL'EDIFICIO DA ANALIZZARE
# L'indice (numero di riga) dell'edificio da analizzare nello shapefile (0 = primo elemento).
TARGET_BUILDING_INDEX = 0 


# ----------------------
# Utility (invariate)
# ----------------------

def extract_zip_find_shp(zip_path: str, out_dir: str):
    """Estrae zip in out_dir e ritorna lista di .shp trovati"""
    shp_list = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    for p in Path(out_dir).rglob('*.shp'):
        shp_list.append(str(p))
    return shp_list


def pick_building_shp(shp_paths):
    """Sceglie il file shp che probabilmente contiene edifici (heuristic).
    Se non trova 'build' nel nome file, ritorna il primo"""
    if not shp_paths:
        raise FileNotFoundError("No .shp files found in the provided ZIP")
    for p in shp_paths:
        if 'build' in Path(p).name.lower():
            return p
    return shp_paths[0]


def lonlat_to_utm_epsg(lon, lat):
    """Calcola EPSG UTM per la posizione (semplificato: nord emisfero)."""
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone  # assumes northern hemisphere
    return epsg


def wrap_to_minus180_180(deg):
    """Wraps an angle in degrees to the range (-180, 180]."""
    # Usiamo (x + 180) % 360 - 180 per un wrapping efficiente
    deg = (deg + 180) % 360 - 180
    # Gestisce il caso limite -180 (lo vogliamo come +180) se necessario
    # Ma per aspect -180 è valido (Nord).
    return deg


def pvgis_aspect_from_azimuth(azimuth):
    """Converts 0-360 azimuth (N=0, E=90) to PVGIS aspect (-180..180, S=0)."""
    # PVGIS aspect = azimuth - 180
    return wrap_to_minus180_180(azimuth - 180)


# ----------------------
# Core (invariate)
# ----------------------

def estimate_height_attr(feat):
    """Tenta estrarre l'altezza da varie proprietà comuni.
    Se fallisce, ritorna DEFAULT_HEIGHT_M
    """
    if isinstance(feat, dict):
        props = feat
    else:
        # geopandas Series behaves like dict for .get
        props = feat
    # Common tags
    keys_try = ['Height', 'building:height', 'bldg:height', 'roof:height', 'levels', 'building:levels', 'floors']
    for k in keys_try:
        if k in props and props.get(k) not in (None, '', 'nan'):
            try:
                val = props.get(k)
                # handle cases like '10 m'
                if isinstance(val, str):
                    val = val.split(' ')[0]
                val = float(val)
                if val <= 0:
                    continue
                if 'level' in k or 'floor' in k:
                    return val * ASSUME_LEVEL_HEIGHT
                return val
            except Exception:
                continue
    return DEFAULT_HEIGHT_M


def estimate_peak_power(geometry):
    """
    Stima la potenza di picco (kWp) in base all'area della geometria in m².
    Formula: Potenza di Picco (kWp) = (Area * 0.4 * 200) / 1000
    """
    if geometry is None or geometry.is_empty:
        return 0.0

    # Gestione MultiPolygon: scegli il più grande (come in orientation)
    if geometry.geom_type == 'MultiPolygon':
        area_m2 = max(p.area for p in geometry.geoms)
    else:
        area_m2 = geometry.area
        
    peakpower_wp = area_m2 * ROOF_AREA_FACTOR * POWER_DENSITY_W_PER_M2
    peakpower_kwp = peakpower_wp / 1000.0
    
    return round(peakpower_kwp, 2), round(area_m2, 2)


def compute_userhorizon_from_gdf(gdf, target_idx, step_deg=10, ray_length=2000):
    """Calcola la lista di angoli (in gradi) per PVGIS.
    gdf deve essere in CRS metrico (metri)
    """
    # Assumiamo che target_idx sia corretto qui.
    target = gdf.iloc[target_idx]
    target_geom = target.geometry
    if not target_geom:
        raise ValueError("Target geometry is empty")

    centroid = target_geom.centroid
    target_height = estimate_height_attr(target)

    # Prepara altezze per tutti gli edifici
    gdf = gdf.copy()
    gdf['__est_h'] = gdf.apply(estimate_height_attr, axis=1)
    
    h_target_calc = gdf.iloc[target_idx]['__est_h']
    target_height = estimate_height_attr(target)


    horizon = []
    angles = np.arange(0, 360, step_deg)

    for angle in angles:
        dx = ray_length * math.sin(math.radians(angle))
        dy = ray_length * math.cos(math.radians(angle))
        ray = LineString([centroid, Point(centroid.x + dx, centroid.y + dy)])
        max_angle = 0.0
        first_intersection_point = None
        first_intersection_height = None

        # Interseca tutti gli edifici
        for idx, row in gdf.iterrows():
            # skip target itself
            if idx == target_idx:
                continue
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            
            # Usiamo 'intersects' prima di 'intersection' per velocità
            if not ray.intersects(geom):
                continue
                
            inter = ray.intersection(geom)
            if inter.is_empty:
                continue
                
            # troviamo il punto di intersezione più vicino al centroide
            try:
                # Gestisce MultiPoint/MultiLineString in 'inter'
                if inter.geom_type == 'GeometryCollection':
                    # prendi solo le parti dimensionalmente valide
                    parts = [p for p in inter.geoms if not p.is_empty]
                    if not parts:
                        continue
                    inter = parts[0] # euristica: prendi il primo
                
                # nearest_points(A, B) ritorna (pt_on_A, pt_on_B)
                np_pair = nearest_points(centroid, inter)
                inter_pt = np_pair[1] # punto sull'ostacolo/intersezione
                
            except Exception as e:
                # fallback:
                # print(f"WARN: nearest_points failed for angle {angle}: {e}")
                inter_pt = geom.representative_point()

            dist = centroid.distance(inter_pt)
            if dist <= 1e-6: # troppo vicino, prob. self-intersection o adiacente
                continue

            h_b = row['__est_h']
            
            # Calcola angolo di elevazione
            h_diff = h_b - target_height
            if h_diff <= 0: # ostacolo più basso del target
                continue

            elev_angle = math.degrees(math.atan(h_diff / dist))
            
            if elev_angle > max_angle:
                max_angle = elev_angle
                first_intersection_point = inter_pt
                first_intersection_height = h_b

        # Non vogliamo valori negativi (ostacoli sotto il livello pannello)
        # Già gestito da h_diff <= 0 check, ma doppia sicurezza
        if max_angle < 0:
            max_angle = 0.0
            
        horizon.append({'angle': int(angle), 'deg': round(max_angle, 2),
                        'pt': first_intersection_point, 'h': first_intersection_height})

    # Formatta come lista di soli angoli (ordinata da 0 a 360-step)
    horizon_degrees = [item['deg'] for item in horizon]
    return horizon, horizon_degrees, centroid, target_height


def compute_panel_orientation(geometry, crs_projected=True):
    """
    Calcola l'orientamento del pannello (azimuth) da una geometria di edificio
    proiettata (metrica) usando la regola MBRect/South-preference.
    
    Args:
        geometry (shapely.geometry): Geometria (Polygon o MultiPolygon) in CRS metrico.
        crs_projected (bool): Flag (attualmente non usato) che assume True.

    Returns:
        dict: Dizionario con i risultati dell'orientamento.
    """
    
    # Dizionario dei risultati, default a incerto/North
    results = {
        'panel_azimuth_deg': 0.0,
        'pvgis_aspect': -180.0,
        'chosen_long_side_midpoint': None,
        'chosen_long_side_endpoints': None,
        'chosen_side_azimuth_deg': 0.0,
        'candidate_azimuths': [0.0, 0.0],
        'candidate_pvgis_aspects': [-180.0, -180.0],
        'chosen_candidate_index': 0,
        'long_side_length': 0.0,
        'short_side_length': 0.0,
        'num_exterior_vertices': 0,
        'uncertain_orientation_flag': True, # Inizia incerto
        'method': 'MBRect_rect_south_midpoint_preference',
        'error': None,
        'mbrect_geom': None, # Per il plotting
        'long_sides_midpoints': [] # Per il plotting
    }

    try:
        if not geometry or geometry.is_empty:
            raise ValueError("Input geometry is empty")

        # 1. Gestione MultiPolygon: scegli il più grande
        if geometry.geom_type == 'MultiPolygon':
            geometry = max(geometry.geoms, key=lambda p: p.area)
        
        if not (geometry.geom_type == 'Polygon' and geometry.exterior):
             raise ValueError(f"Geometry is not a valid Polygon (type: {geometry.geom_type})")

        # 2. Input è già proiettato (assunto)
        
        results['num_exterior_vertices'] = len(geometry.exterior.coords) - 1

        # 3. Calcola MBRect
        mbrect = geometry.minimum_rotated_rectangle
        results['mbrect_geom'] = mbrect
        if mbrect.is_empty or not mbrect.exterior:
            raise ValueError("MBRect is empty or invalid")

        # 4. Estrai 4 vertici unici
        coords = list(mbrect.exterior.coords)
        unique_vertices = coords[:-1]
        
        if len(unique_vertices) != 4:
            raise ValueError(f"MBRect has {len(unique_vertices)} unique vertices, expected 4.")

        # 5. Calcola segmenti e lunghezze
        sides = []
        for i in range(4):
            p1 = unique_vertices[i]
            p2 = unique_vertices[(i + 1) % 4]
            length = math.dist(p1, p2) # math.dist richiede Python 3.8+
            sides.append({'p1': p1, 'p2': p2, 'length': length})
        
        sides.sort(key=lambda s: s['length'], reverse=True)
        
        long_side_1 = sides[0]
        long_side_2 = sides[1]
        long_side_length = sides[0]['length']
        short_side_length = sides[2]['length']
        results['long_side_length'] = round(long_side_length, 2)
        results['short_side_length'] = round(short_side_length, 2)

        if long_side_length < 1e-6:
            raise ValueError("MBRect is degenerate (zero length)")

        # 6. Identifica i 2 lati lunghi e scegli quello a SUD
        
        # Calcola midpoints (usando la media, più veloce di .centroid)
        midpoint_1_coords = ((long_side_1['p1'][0] + long_side_1['p2'][0]) / 2,
                             (long_side_1['p1'][1] + long_side_1['p2'][1]) / 2)
        midpoint_2_coords = ((long_side_2['p1'][0] + long_side_2['p2'][0]) / 2,
                             (long_side_2['p1'][1] + long_side_2['p2'][1]) / 2)
        
        results['long_sides_midpoints'] = [Point(midpoint_1_coords), Point(midpoint_2_coords)]

        # Scegli lato con midpoint y minore (più a sud)
        # Tie-breaker: x minore
        if midpoint_1_coords[1] < midpoint_2_coords[1]:
            chosen_side = long_side_1
        elif midpoint_2_coords[1] < midpoint_1_coords[1]:
            chosen_side = long_side_2
        else: # y uguali
            chosen_side = long_side_1 if midpoint_1_coords[0] <= midpoint_2_coords[0] else long_side_2

        results['chosen_long_side_midpoint'] = ((chosen_side['p1'][0] + chosen_side['p2'][0]) / 2,
                                                (chosen_side['p1'][1] + chosen_side['p2'][1]) / 2)
        results['chosen_long_side_endpoints'] = [chosen_side['p1'], chosen_side['p2']]

        # 7. Vettore del lato scelto
        p1, p2 = chosen_side['p1'], chosen_side['p2']
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # 8. Azimuth del lato (da Nord, N=0, E=90)
        az_side = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        results['chosen_side_azimuth_deg'] = round(az_side, 2)

        # 9. Azimuth candidati (perpendicolari)
        cand0 = (az_side + 90) % 360
        cand1 = (az_side - 90 + 360) % 360
        results['candidate_azimuths'] = [round(cand0, 2), round(cand1, 2)]

        # 10. Calcola PVGIS aspect per i candidati
        pvgis0 = pvgis_aspect_from_azimuth(cand0)
        pvgis1 = pvgis_aspect_from_azimuth(cand1)
        results['candidate_pvgis_aspects'] = [round(pvgis0, 2), round(pvgis1, 2)]
        
        # 11. Scegli il candidato più vicino a Sud (min(abs(aspect)))
        if abs(pvgis0) <= abs(pvgis1):
            # Scegli cand0. Il <= gestisce il tie-break (es. 90 vs 270)
            results['chosen_candidate_index'] = 0
        else:
            results['chosen_candidate_index'] = 1
            
        results['panel_azimuth_deg'] = results['candidate_azimuths'][results['chosen_candidate_index']]
        results['pvgis_aspect'] = results['candidate_pvgis_aspects'][results['chosen_candidate_index']]

        # 12. Regole di incertezza
        uncertain = False
        if results['num_exterior_vertices'] > 4:
            uncertain = True
        elif short_side_length > 1e-6:
             if (long_side_length / short_side_length) < ORIENTATION_UNCERTAINTY_RATIO:
                 uncertain = True # near-square
        else:
             uncertain = True # degenerate

        results['uncertain_orientation_flag'] = uncertain
        
    except Exception as e:
        results['error'] = str(e)
        results['uncertain_orientation_flag'] = True # Errore = incerto
    
    return results


# ----------------------
# PVGIS call (MODIFICATA)
# ----------------------

def call_pvgis_seriescalc(lat, lon, userhorizon_str, peakpower, tilt=None, aspect=None):
    """
    Chiamata API PVGIS v5_2 (seriescalc).
    Ritorna il JSON completo.
    [MODIFICA]: Aggiunge startyear e endyear per limitare l'output a un solo anno (8760 righe).
    """
    # Se tilt non fornito, usa il default
    if tilt is None:
        tilt = DEFAULT_TILT_FOR_ASPECT
    
    # Se aspect non fornito, usa 0 (Sud)
    if aspect is None:
        aspect = 0

    params = {
        'lat': lat,
        'lon': lon,
        'peakpower': peakpower,
        # Parametri seriescalc
        'pvcalculation': 1,  # Per dati orari
        'outputformat': 'json',
        'browser': 0,
        # *** FIX PER TMY/ANNO SINGOLO ***
        # Forza l'output a un solo anno (8760 righe) per simulare un anno tipico (TMY)
        # La vera TMY richiederebbe l'endpoint /tmy
        'startyear': 2020,
        'endyear': 2020,
        # *******************************
        # Parametri PVcalc
        'usehorizon': 1,
        'userhorizon': userhorizon_str,
        'mountingplace': 'free',
        'loss': 0.0,
        # Parametri fissi
        'optimalinclination': 0, # Disabilita ottimizzazione inclinazione
        'optimalangles': 0,      # Disabilita ottimizzazione orientamento
        'tilt': tilt,
        'aspect': aspect
    }

    resp = requests.get(PVGIS_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ----------------------
# Plot (invariata rispetto all'ultimo miglioramento)
# ----------------------

def plot_scene(gdf, target_idx, centroid, horizon_items, orientation_results, out_png=PLOT_OUTPUT):
    """
    Salva un plot della scena, includendo i risultati dell'orientamento.
    """
    # [MODIFICA 1]: Aumento le dimensioni della figura per dare più spazio alle annotazioni
    fig, ax = plt.subplots(figsize=(15,15)) # 15x15 invece di 20x20
    
    # Impostazioni di base del plot
    base = gdf.plot(ax=ax, color='lightgrey', edgecolor='gray', alpha=0.7)
    target = gdf.iloc[target_idx:target_idx+1]
    target.plot(ax=base, color='salmon', edgecolor='red')

    # plot centroid
    ax.scatter([centroid.x], [centroid.y], color='red', zorder=5)

    # --- Plot Orizzonte ---
    # Definisco una dimensione del carattere base più leggibile
    FONT_SIZE_HORIZON = 12 
    
    for item in horizon_items:
        angle = item['angle']
        dx_r = RAY_LENGTH_M * math.sin(math.radians(angle))
        dy_r = RAY_LENGTH_M * math.cos(math.radians(angle))
        x_end = centroid.x + dx_r
        y_end = centroid.y + dy_r
        ax.plot([centroid.x, x_end], [centroid.y, y_end], 
                linestyle='--', color='gray', linewidth=0.8, alpha=0.6)

        if item['pt'] is not None:
            ip = item['pt']
            ax.scatter([ip.x], [ip.y], marker='x', color='blue', s=40)
            txt = f"{item['deg']}°"
            ax.text(ip.x, ip.y, txt, fontsize=FONT_SIZE_HORIZON, color='blue', ha='left', va='bottom')
        else:
            # segna la distanza massima lungo il raggio
            ax.text(x_end, y_end, f"{item['deg']}°", fontsize=FONT_SIZE_HORIZON*0.8, alpha=0.6, color='gray', ha='center', va='center')

    # --- Plot Orientamento ---
    try:
        # Plot MBRect
        mbrect_geom = orientation_results.get('mbrect_geom')
        if mbrect_geom:
            gpd.GeoSeries([mbrect_geom], crs=gdf.crs).plot(
                ax=base, facecolor='none', edgecolor='blue', linestyle=':', 
                linewidth=1.5, label='MBRect')

        # Plot midpoints lati lunghi
        midpoints = orientation_results.get('long_sides_midpoints', [])
        chosen_midpoint_coord = orientation_results.get('chosen_long_side_midpoint')
        
        if midpoints:
            other_midpoints_pts = []
            for pt in midpoints:
                # Confronta le coordinate
                if chosen_midpoint_coord and math.isclose(pt.x, chosen_midpoint_coord[0]) and math.isclose(pt.y, chosen_midpoint_coord[1]):
                    continue
                other_midpoints_pts.append(pt)
            if other_midpoints_pts:
                ax.scatter([pt.x for pt in other_midpoints_pts], [pt.y for pt in other_midpoints_pts],
                           color='blue', marker='o', s=50, alpha=0.8,
                           label='Other long side midpoint')
        
        if chosen_midpoint_coord:
            ax.scatter([chosen_midpoint_coord[0]], [chosen_midpoint_coord[1]],
                       color='cyan', marker='*', s=250, edgecolor='black', zorder=10,
                       label='Chosen long side midpoint (South-most)')

        # Plot freccia Azimuth pannello
        panel_az = orientation_results.get('panel_azimuth_deg')
        pvgis_asp = orientation_results.get('pvgis_aspect')
        
        if panel_az is not None:
            # Lunghezza freccia
            arrow_len = min(RAY_LENGTH_M * 0.3, max(orientation_results.get('long_side_length', 50), 50))
            
            # [MODIFICA 2]: Ridimensiono la testa della freccia
            head_size_ratio = 0.08 
            dx_a = arrow_len * math.sin(math.radians(panel_az))
            dy_a = arrow_len * math.cos(math.radians(panel_az))
            
            ax.arrow(centroid.x, centroid.y, dx_a, dy_a,
                     head_width=arrow_len*head_size_ratio, head_length=arrow_len*head_size_ratio*1.5,
                     fc='magenta', ec='magenta', zorder=9, lw=2.5,
                     label='Pannello Azimuth')
            
            # [MODIFICA 3]: Aumento la dimensione del testo e lo sposto leggermente
            ax.text(centroid.x + dx_a * 1.15, centroid.y + dy_a * 1.15, 
                    f"Panel Az: {panel_az}°\n(Aspect: {pvgis_asp}°)",
                    color='magenta', fontsize=FONT_SIZE_HORIZON+2, ha='center', va='center', weight='bold')

        # Plot label Incertezza
        if orientation_results.get('uncertain_orientation_flag'):
            # [MODIFICA 4]: Aumento dimensione testo e bordo del box più discreto
            ax.text(centroid.x, centroid.y,
                    "UNCERTAIN ORIENTATION\n(Used MBRect rule as fallback)",
                    color='red', fontsize=FONT_SIZE_HORIZON+1, ha='center', va='bottom', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', linewidth=0.5, boxstyle='round,pad=0.5'),
                    zorder=11)

    except Exception as e:
        print(f"Warning: Failed to plot orientation details. {e}")
    # --- Fine Plot Orientamento ---

    # [MODIFICA 5]: Titolo e Assi più grandi
    ax.set_title(f'Buildings, Horizon (step {STEP_DEG}°) and Orientation (Target Idx {target_idx})', fontsize=FONT_SIZE_HORIZON+4)
    ax.set_xlabel('x (m)', fontsize=FONT_SIZE_HORIZON+2)
    ax.set_ylabel('y (m)', fontsize=FONT_SIZE_HORIZON+2)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_HORIZON) # Aumento dimensione etichette assi
    
    # [MODIFICA 6]: Ottimizzazione dello zoom: include raggi e buffer
    minx, miny, maxx, maxy = target.total_bounds
    
    # La lunghezza massima è RAY_LENGTH_M. Lo zoom deve coprire l'edificio target
    buffer_zoom = RAY_LENGTH_M * 1.1 
    
    # Calcolo dei limiti basato sul centroide e sul buffer
    center_x, center_y = centroid.x, centroid.y
    
    # Determino l'estensione massima necessaria e la applico simmetricamente dal centroide
    x_min_new = min(gdf.total_bounds[0], centroid.x - buffer_zoom)
    y_min_new = min(gdf.total_bounds[1], centroid.y - buffer_zoom)
    x_max_new = max(gdf.total_bounds[2], centroid.x + buffer_zoom)
    y_max_new = max(gdf.total_bounds[3], centroid.y + buffer_zoom)
    
    max_extent = max(x_max_new - x_min_new, y_max_new - y_min_new) * 1.05 # 5% in più per i margini
    half_max_extent = max_extent / 2.0
    
    ax.set_xlim(center_x - half_max_extent, center_x + half_max_extent)
    ax.set_ylim(center_y - half_max_extent, center_y + half_max_extent)
    
    
    legend_elements = [
    Line2D([0], [0], marker='s', color='w', label='Buildings',
           markerfacecolor='gray', markersize=8),
    Line2D([0], [0], marker='s', color='w', label='Target Building',
           markerfacecolor='salmon', markeredgecolor='red', markersize=8)
]

# Applica la legenda usando gli elementi fittizi
    ax.legend(handles=legend_elements, loc='best', fontsize=FONT_SIZE_HORIZON)
    
    # [MODIFICA 7]: Usa bbox_inches='tight' in savefig 
    plt.savefig(out_png, dpi=400, bbox_inches='tight') 
    print(f"\nPlot saved to: {out_png}")
    plt.close(fig)


# ----------------------
# Test function (invariata)
# ----------------------

def run_orientation_tests():
    """Esegue test di validazione per la funzione compute_panel_orientation."""
    print("--- Running Orientation Tests ---")
    
    # Test 1: Rettangolo E-W (lati lunghi E-W)
    print("\nTest 1: E-W Rectangle (10x4)")
    poly1 = Polygon([(0,0),(10,0),(10,4),(0,4)])
    res1 = compute_panel_orientation(poly1)
    print(f"  Result: Az={res1['panel_azimuth_deg']}, Aspect={res1['pvgis_aspect']}, Uncertain={res1['uncertain_orientation_flag']}")
    print(f"  Details: SideAz={res1['chosen_side_azimuth_deg']}, Cands={res1['candidate_azimuths']}, Idx={res1['chosen_candidate_index']}")
    assert res1['panel_azimuth_deg'] == 180.0
    assert res1['pvgis_aspect'] == 0.0
    assert not res1['uncertain_orientation_flag']
    assert math.isclose(res1['chosen_side_azimuth_deg'], 90.0)
    assert res1['chosen_long_side_midpoint'] == (5.0, 0.0)

    # Test 2: Rettangolo N-S (lati lunghi N-S) + Tie-break
    print("\nTest 2: N-S Rectangle (4x10) + Tie-Break Test")
    poly2 = Polygon([(0,0),(4,0),(4,10),(0,10)])
    res2 = compute_panel_orientation(poly2)
    print(f"  Result: Az={res2['panel_azimuth_deg']}, Aspect={res2['pvgis_aspect']}, Uncertain={res2['uncertain_orientation_flag']}")
    print(f"  Details: SideAz={res2['chosen_side_azimuth_deg']}, Cands={res2['candidate_azimuths']}, Idx={res2['chosen_candidate_index']}")
    assert res2['chosen_long_side_midpoint'] == (0.0, 5.0) # Midpoint con x minore
    assert math.isclose(res2['chosen_side_azimuth_deg'], 0.0) or math.isclose(res2['chosen_side_azimuth_deg'], 180.0)
    assert res2['panel_azimuth_deg'] == 90.0 # Sceglie Cand[0] (90)
    assert res2['pvgis_aspect'] == -90.0
    assert not res2['uncertain_orientation_flag']

    # Test 3: Near-square (ratio < 1.05)
    print("\nTest 3: Near-Square (10x9.8)")
    poly3 = Polygon([(0,0),(10,0),(10,9.8),(0,9.8)])
    res3 = compute_panel_orientation(poly3)
    print(f"  Result: Az={res3['panel_azimuth_deg']}, Aspect={res3['pvgis_aspect']}, Uncertain={res3['uncertain_orientation_flag']}")
    print(f"  Ratio: {res3['long_side_length'] / res3['short_side_length']:.4f}")
    assert res3['uncertain_orientation_flag'] # Deve essere incerto
    assert res3['panel_azimuth_deg'] == 180.0 # Ma calcola comunque (come Test 1)

    # Test 4: >4 vertici
    print("\nTest 4: Complex Polygon (>4 vertices)")
    poly4 = Polygon([(0,0),(10,0),(10,4),(5,6),(0,4)])
    res4 = compute_panel_orientation(poly4)
    print(f"  Result: Az={res4['panel_azimuth_deg']}, Aspect={res4['pvgis_aspect']}, Uncertain={res4['uncertain_orientation_flag']}")
    print(f"  Vertices: {res4['num_exterior_vertices']}")
    assert res4['uncertain_orientation_flag']
    assert res4['num_exterior_vertices'] == 5
    # MBRect è (0,0)-(10,0)-(10,6)-(0,6). Lato lungo E-W.
    # Scelto lato sud. Az 90. Cand [180, 0]. Scelto 180.
    assert res4['panel_azimuth_deg'] == 180.0 
    assert res4['pvgis_aspect'] == 0.0

    print("\n--- All Tests Passed ---")


# ----------------------
# Main (invariata rispetto all'ultimo FIX)
# ----------------------

def main(zip_path):
    tmpdir = tempfile.mkdtemp(prefix='pv_horizon_')
    summary_data = {} # Per JSON output
    
    # 🎯 USA LA COSTANTE GLOBALE QUI
    target_idx = TARGET_BUILDING_INDEX
    
    try:
        shp_paths = extract_zip_find_shp(zip_path, tmpdir)
        shp = pick_building_shp(shp_paths)
        print(f"Using shapefile: {shp}")

        gdf = gpd.read_file(shp)

        if gdf.empty:
            raise ValueError("The building layer is empty")
        
        # Verifica se l'indice è fuori range
        if target_idx < 0 or target_idx >= len(gdf):
             raise IndexError(f"TARGET_BUILDING_INDEX={target_idx} is out of bounds for the shapefile with {len(gdf)} buildings (max index is {len(gdf)-1}).")

        if gdf.crs is None:
            print("Input layer has no CRS — assuming EPSG:4326 (lon/lat). If this is incorrect, reproject accordingly.")
            gdf = gdf.set_crs(epsg=4326)

        # get a representative lon/lat
        gdf_ll = gdf.to_crs(epsg=4326)
        # Usiamo il centroide dell'intero layer per il CRS (più robusto)
        rep_point = gdf_ll.union_all().centroid
        lon, lat = rep_point.x, rep_point.y
        utm_epsg = lonlat_to_utm_epsg(lon, lat)
        print(f"Reprojecting geometries to UTM EPSG:{utm_epsg} for metric calculations")
        gdf = gdf.to_crs(epsg=utm_epsg)

        # --- Calcolo Orizzonte ---
        print(f"Computing horizon for target building (index {target_idx})")
        horizon_items, horizon_degrees, centroid, target_height = compute_userhorizon_from_gdf(
            gdf, target_idx=target_idx, step_deg=STEP_DEG, ray_length=RAY_LENGTH_M)

        userhorizon_str = ','.join(str(d) for d in horizon_degrees)
        print('\nUSERHORIZON (PVGIS format):')
        print(userhorizon_str)
        
        # --- Calcolo Orientamento ---
        print("\nComputing orientation for target building...")
        target_geom = gdf.iloc[target_idx].geometry
        orientation_results = compute_panel_orientation(target_geom)
        
        print('\nORIENTATION SUMMARY:')
        print(f"  panel_azimuth_deg: {orientation_results['panel_azimuth_deg']}")
        print(f"  pvgis_aspect: {orientation_results['pvgis_aspect']}")
        print(f"  chosen_long_side_midpoint: {orientation_results['chosen_long_side_midpoint']}")
        print(f"  uncertain_orientation_flag: {orientation_results['uncertain_orientation_flag']}")
        
        # --- [NUOVO] Calcolo Potenza di Picco ---
        peakpower_kwp, area_m2 = estimate_peak_power(target_geom)
        print('\nPEAKPOWER SUMMARY:')
        print(f"  Building Area: {area_m2} m²")
        print(f"  Estimated Peak Power: {peakpower_kwp} kWp (Area Factor {ROOF_AREA_FACTOR}, Density {POWER_DENSITY_W_PER_M2} W/m²)")
        
        # --- Plot ---
        plot_scene(gdf, target_idx, centroid, horizon_items, orientation_results, out_png=PLOT_OUTPUT)


        # --- [MODIFICATO] Chiamata PVGIS seriescalc ---
        # Converti centroide target (proiettato) in lat/lon
        centroid_ll = gpd.GeoSeries([centroid], crs=f"EPSG:{utm_epsg}").to_crs(epsg=4326).iloc[0]
        lat_pt = centroid_ll.y
        lon_pt = centroid_ll.x
        
        # Usiamo l'aspect calcolato
        pvgis_aspect = orientation_results['pvgis_aspect']
        
        # Ora la chiamata userà solo l'anno 2020 (8760 righe)
        print(f"\nCalling PVGIS seriescalc for lat={lat_pt:.6f}, lon={lon_pt:.6f} (peakpower={peakpower_kwp} kWp)")
        print(f"  Using calculated aspect={pvgis_aspect:.1f} (Tilt={DEFAULT_TILT_FOR_ASPECT}° [default])")

        try:
            pvgis_json = call_pvgis_seriescalc(lat_pt, lon_pt, 
                                               userhorizon_str, 
                                               peakpower=peakpower_kwp, 
                                               aspect=pvgis_aspect)
            
            print('\nPVGIS seriescalc response keys:', list(pvgis_json.keys()))
            
            # --- [FIX & NUOVO] Elaborazione e Output PVGIS ---
            if 'outputs' in pvgis_json and 'hourly' in pvgis_json['outputs']:
                
                hourly_data = pvgis_json['outputs']['hourly']
                df = pd.DataFrame(hourly_data)
                
                # Calcola Energia Totale Annua (kWh)
                # La potenza P è in W. La somma è in Wh. / 1000 per kWh.
                # Per la serie oraria annuale (8760 ore)
                total_annual_energy_kwh = (df['P'].sum() / 1000.0)
                
                print('\nPVGIS Total Annual Energy:')
                print(f"  Total Annual Energy (P, Wh/1000): {total_annual_energy_kwh:.2f} kWh")
                
                # Salva i dati orari in un file CSV separato
                # [FIX]: Seleziona solo le colonne che sono effettivamente presenti
                expected_cols = ['time', 'P', 'Gb(i)', 'Gd(i)', 'Gr(i)', 'H_sun', 'T2m']
                available_cols = [col for col in expected_cols if col in df.columns]
                
                df_out = df[available_cols]
                df_out.to_csv(HOURLY_OUTPUT, index=False)
                print(f"Hourly series saved to: {HOURLY_OUTPUT}")

                # Aggiorna il JSON di riepilogo
                summary_data['pvgis_results'] = {
                    'total_annual_energy_kwh': round(total_annual_energy_kwh, 2),
                    'peakpower_used_kwp': peakpower_kwp,
                    'tseries_info': {
                        'start': df['time'].iloc[0],
                        'end': df['time'].iloc[-1],
                        'count': len(df)
                    },
                    'data_source': pvgis_json.get('meta', {}).get('radiation_db', 'N/A')
                }
                
            else:
                print('\nPVGIS returned (raw):')
                print(json.dumps(pvgis_json, indent=2)[:1000])
                summary_data['pvgis_results'] = pvgis_json

        except Exception as e:
            # Cattura l'errore specifico del 'not in index' che ora dovrebbe essere risolto,
            # ma anche eventuali altri errori.
            print('PVGIS seriescalc request failed:', e)
            summary_data['pvgis_results'] = {'error': str(e)}

        # --- Salva Summary JSON ---
        
        # Pulisci i risultati dell'orientamento per JSON (rimuovi geometrie)
        orientation_for_json = orientation_results.copy()
        orientation_for_json.pop('mbrect_geom', None)
        orientation_for_json.pop('long_sides_midpoints', None)
        
        summary_data['location'] = {
            'centroid_lat_epsg4326': lat_pt,
            'centroid_lon_epsg4326': lon_pt,
            'projected_centroid_xy': (centroid.x, centroid.y),
            'projected_crs': f"EPSG:{utm_epsg}"
        }
        summary_data['target_properties'] = {
            'area_m2': area_m2,
            'target_height_m': target_height,
            'peakpower_kwp_estimate': peakpower_kwp,
            'orientation': orientation_for_json
        }
        summary_data['horizon'] = {
            'userhorizon_string': userhorizon_str,
            'horizon_degrees': horizon_degrees
        }
        
        with open(SUMMARY_OUTPUT, 'w') as f:
            # Funzione di helper per convertire tipi numpy se presenti
            def json_converter(o):
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return str(o) # fallback

            json.dump(summary_data, f, indent=2, default=json_converter)
        print(f"Summary saved to: {SUMMARY_OUTPUT}")


    finally:
        # Pulisci la directory temporanea
        try:
            shutil.rmtree(tmpdir)
            # print(f"Cleaned up temp dir: {tmpdir}")
        except Exception as e:
            print(f"Warning: could not clean up temp dir {tmpdir}. {e}")
        pass


if __name__ == '__main__':
    
    if '--run-orientation-tests' in sys.argv:
        run_orientation_tests()
        sys.exit(0)
    
    if len(sys.argv) < 2:
        print('Usage: python pvgis_horizon_from_shapefile.py path/to/buildings.zip')
        print('   or: python pvgis_horizon_from_shapefile.py --run-orientation-tests')
        sys.exit(1)
        
    zip_path = sys.argv[1]
    if not os.path.isfile(zip_path):
        print('ZIP file not found:', zip_path)
        sys.exit(1)
        
    main(zip_path)