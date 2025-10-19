#!/usr/bin/env python
"""
pvgis_analyzer.py

Modulo di analisi per risultati PVGIS. Gestisce:
- Metriche annuali
- Giorni best/worst
- Sensibilità tilt (on-demand, 5 call PVGIS)
- Impatto horizon (on-demand, 1 call PVGIS)
- Export GeoJSON per Leaflet

Filosofia: Funzioni semplici, lazy loading, storage in-memory.
"""

import math
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# Import dalle funzioni di pvgis_horizon_from_shapefile
# (Adatteremo questi import in base a come rifattorezzi quel file)
import sys
sys.path.insert(0, str(Path(__file__).parent))

from pvgis_horizon_from_shapefile import (
    compute_userhorizon_from_gdf,
    compute_panel_orientation,
    estimate_peak_power,
    estimate_height_attr,
    lonlat_to_utm_epsg,
    call_pvgis_seriescalc,
    extract_zip_find_shp,
    pick_building_shp,
    STEP_DEG,
    RAY_LENGTH_M,
    DEFAULT_HEIGHT_M,
    DEFAULT_TILT_FOR_ASPECT,
)


# ============================================================
# METRICHE ANNUALI
# ============================================================

def compute_annual_metrics(df_hourly: pd.DataFrame, peakpower_kwp: float) -> dict:
    """
    Calcola metriche energetiche annuali da serie oraria PVGIS.
    
    Args:
        df_hourly: DataFrame con colonna 'P' (potenza in W)
        peakpower_kwp: Potenza nominale installata (kWp)
    
    Returns:
        dict con metriche annuali
    """
    if df_hourly.empty or 'P' not in df_hourly.columns:
        return {
            'energy_kwh': 0.0,
            'capacity_factor': 0.0,
            'specific_yield_kwh_kw': 0.0,
            'avg_power_w': 0.0,
            'max_power_w': 0.0,
            'min_power_w': 0.0,
            'peak_hours_h': 0.0,
            'num_hours': 0
        }
    
    power_w = df_hourly['P'].values
    
    # Energia totale annua (Wh → kWh)
    energy_wh = power_w.sum()
    energy_kwh = energy_wh / 1000.0
    
    # Potenza media/max/min
    avg_power_w = float(np.mean(power_w))
    max_power_w = float(np.max(power_w))
    min_power_w = float(np.min(power_w))
    
    # Capacity Factor = Energia reale / (Potenza nominale × 8760 ore)
    nominal_annual_kwh = peakpower_kwp * 8760
    capacity_factor = energy_kwh / nominal_annual_kwh if nominal_annual_kwh > 0 else 0.0
    
    # Produttività specifica (kWh/kW)
    specific_yield = energy_kwh / peakpower_kwp if peakpower_kwp > 0 else 0.0
    
    # Ore equivalenti a potenza nominale (kWh / kWp)
    peak_hours = specific_yield
    
    return {
        'energy_kwh': round(energy_kwh, 2),
        'capacity_factor': round(capacity_factor, 4),
        'specific_yield_kwh_kw': round(specific_yield, 2),
        'avg_power_w': round(avg_power_w, 2),
        'max_power_w': round(max_power_w, 2),
        'min_power_w': round(min_power_w, 2),
        'peak_hours_h': round(peak_hours, 2),
        'num_hours': len(df_hourly)
    }


# ============================================================
# GIORNI BEST & WORST
# ============================================================

def compute_best_worst_days(df_hourly: pd.DataFrame) -> dict:
    """
    Trova i giorni con massima e minima energia.
    
    Args:
        df_hourly: DataFrame con colonne 'time' (str YYYYMMDD:HHMM da PVGIS) e 'P' (W)
    
    Returns:
        dict con profili orari best/worst
    """
    if df_hourly.empty or 'P' not in df_hourly.columns or 'time' not in df_hourly.columns:
        return {
            'best': {'date': None, 'energy_kwh': 0.0, 'profile': []},
            'worst': {'date': None, 'energy_kwh': 0.0, 'profile': []}
        }
    
    # Parse time: PVGIS format è YYYYMMDD:HHMM (e.g., "20200101:0011")
    df_hourly = df_hourly.copy()
    try:
        df_hourly['datetime'] = pd.to_datetime(df_hourly['time'], format='%Y%m%d:%H%M')
    except Exception as e:
        print(f"Warning: Could not parse time with format '%Y%m%d:%H%M'. Trying alternative formats. Error: {e}")
        try:
            df_hourly['datetime'] = pd.to_datetime(df_hourly['time'])
        except:
            print(f"ERROR: Could not parse time column at all")
            return {
                'best': {'date': None, 'energy_kwh': 0.0, 'profile': []},
                'worst': {'date': None, 'energy_kwh': 0.0, 'profile': []}
            }
    
    df_hourly['date'] = df_hourly['datetime'].dt.date
    
    # Raggruppa per giorno e somma energia
    daily_energy = df_hourly.groupby('date')['P'].agg(['sum', 'count'])
    daily_energy.columns = ['energy_wh', 'count']
    daily_energy['energy_kwh'] = daily_energy['energy_wh'] / 1000.0
    
    # Filtra giorni completi (24 ore)
    daily_energy = daily_energy[daily_energy['count'] == 24]
    
    if daily_energy.empty:
        return {
            'best': {'date': None, 'energy_kwh': 0.0, 'profile': []},
            'worst': {'date': None, 'energy_kwh': 0.0, 'profile': []}
        }
    
    # Best e worst day
    best_date = daily_energy['energy_kwh'].idxmax()
    worst_date = daily_energy['energy_kwh'].idxmin()
    
    best_energy = daily_energy.loc[best_date, 'energy_kwh']
    worst_energy = daily_energy.loc[worst_date, 'energy_kwh']
    
    # Estrai profili orari
    best_profile = df_hourly[df_hourly['date'] == best_date].sort_values('datetime')['P'].tolist()
    worst_profile = df_hourly[df_hourly['date'] == worst_date].sort_values('datetime')['P'].tolist()
    
    return {
        'best': {
            'date': str(best_date),
            'energy_kwh': round(best_energy, 2),
            'profile': [round(p, 2) for p in best_profile]
        },
        'worst': {
            'date': str(worst_date),
            'energy_kwh': round(worst_energy, 2),
            'profile': [round(p, 2) for p in worst_profile]
        }
    }


# ============================================================
# SENSIBILITÀ TILT (On-Demand)
# ============================================================

def compute_tilt_sensitivity(
    lat: float,
    lon: float,
    userhorizon_str: str,
    aspect_deg: float,
    peakpower_kwp: float,
    tilt_values: list = None
) -> dict:
    """
    Calcola energia per diversi valori di tilt.
    Richiede N call a PVGIS.
    
    Args:
        lat, lon: Coordinate
        userhorizon_str: Stringa horizon per PVGIS
        aspect_deg: Aspetto del pannello (PVGIS format)
        peakpower_kwp: Potenza nominale
        tilt_values: Lista di tilt da testare [default: [15, 20, 25, 30, 35]]
    
    Returns:
        dict con risultati sensibilità e tilt ottimale
    """
    if tilt_values is None:
        tilt_values = [15, 20, 25, 30, 35]
    
    tilt_values = sorted(tilt_values)
    energy_values = []
    
    print(f"\n[TILT SENSITIVITY] Computing {len(tilt_values)} scenarios...")
    
    for tilt in tilt_values:
        print(f"  - Tilt {tilt}°...", end=' ', flush=True)
        try:
            pvgis_json = call_pvgis_seriescalc(
                lat, lon,
                userhorizon_str,
                peakpower_kwp,
                tilt=tilt,
                aspect=aspect_deg
            )
            
            if 'outputs' in pvgis_json and 'hourly' in pvgis_json['outputs']:
                df = pd.DataFrame(pvgis_json['outputs']['hourly'])
                energy_kwh = (df['P'].sum() / 1000.0)
                energy_values.append(energy_kwh)
                print(f"{energy_kwh:.0f} kWh")
            else:
                energy_values.append(0.0)
                print("ERROR")
        except Exception as e:
            print(f"FAILED: {e}")
            energy_values.append(0.0)
    
    # Trova tilt ottimale
    if energy_values and any(e > 0 for e in energy_values):
        optimal_idx = np.argmax(energy_values)
        optimal_tilt = tilt_values[optimal_idx]
        optimal_energy = energy_values[optimal_idx]
    else:
        optimal_tilt = tilt_values[len(tilt_values)//2] if tilt_values else 25
        optimal_energy = 0.0
    
    return {
        'tilt_values': tilt_values,
        'energy_values': [round(e, 2) for e in energy_values],
        'optimal_tilt': optimal_tilt,
        'optimal_energy_kwh': round(optimal_energy, 2),
        'computed': True
    }


# ============================================================
# IMPATTO HORIZON (On-Demand)
# ============================================================

def compute_horizon_impact(
    lat: float,
    lon: float,
    userhorizon_str: str,
    aspect_deg: float,
    peakpower_kwp: float,
    tilt_deg: float,
    energy_with_horizon_kwh: float
) -> dict:
    """
    Calcola impatto dell'ombreggiamento rispetto a scenario piatto.
    Richiede 1 call aggiuntiva a PVGIS (con horizon=0).
    
    Args:
        lat, lon: Coordinate
        userhorizon_str: Stringa horizon corrente
        aspect_deg: Aspetto pannello
        peakpower_kwp: Potenza nominale
        tilt_deg: Inclinazione
        energy_with_horizon_kwh: Energia con ombreggiamento (riferimento)
    
    Returns:
        dict con confronto con/senza horizon
    """
    print(f"\n[HORIZON IMPACT] Computing scenario without shadows...")
    
    # Crea horizon piatto (tutti 0)
    horizon_flat = ','.join(['0'] * 36)  # 10° step → 36 valori
    
    try:
        pvgis_json = call_pvgis_seriescalc(
            lat, lon,
            horizon_flat,
            peakpower_kwp,
            tilt=tilt_deg,
            aspect=aspect_deg
        )
        
        if 'outputs' in pvgis_json and 'hourly' in pvgis_json['outputs']:
            df = pd.DataFrame(pvgis_json['outputs']['hourly'])
            energy_without_kwh = (df['P'].sum() / 1000.0)
        else:
            energy_without_kwh = energy_with_horizon_kwh
    except Exception as e:
        print(f"ERROR in horizon impact calculation: {e}")
        energy_without_kwh = energy_with_horizon_kwh
    
    loss_abs = energy_without_kwh - energy_with_horizon_kwh
    loss_pct = (loss_abs / energy_without_kwh * 100) if energy_without_kwh > 0 else 0.0
    
    return {
        'energy_with_horizon_kwh': round(energy_with_horizon_kwh, 2),
        'energy_without_horizon_kwh': round(energy_without_kwh, 2),
        'loss_abs_kwh': round(loss_abs, 2),
        'loss_pct': round(loss_pct, 2),
        'computed': True
    }


# ============================================================
# PROCESSING SINGOLO EDIFICIO
# ============================================================

def process_building(
    gdf,
    building_idx: int,
    utm_epsg: int = None,
    compute_horizon_impact_flag: bool = False,
    existing_results: dict = None
) -> dict:
    """
    Processa UN edificio: horizon, orientation, PVGIS call, metriche.
    
    Args:
        gdf: GeoDataFrame (in CRS metrico UTM)
        building_idx: Indice dell'edificio
        utm_epsg: EPSG code (se None, calcola da centroid)
        compute_horizon_impact_flag: Se True, calcola impatto horizon (+1 call)
        existing_results: dict esistente per cache check (evita riprocessare)
    
    Returns:
        dict con risultati completi dell'edificio, o None se già processato
    """
    if building_idx < 0 or building_idx >= len(gdf):
        raise IndexError(f"Building index {building_idx} out of range [0, {len(gdf)-1}]")
    
    # CHECK CACHE: Se risultato esiste già, skip
    if existing_results and building_idx in existing_results and existing_results[building_idx] is not None:
        print(f"--- Building {building_idx} (cached, skipping) ---")
        return existing_results[building_idx]
    
    print(f"\n--- Processing Building {building_idx} ---")
    
    # 1. Calcola horizon e orientation
    print("  Computing horizon...")
    horizon_items, horizon_degrees, centroid, target_height = compute_userhorizon_from_gdf(
        gdf, target_idx=building_idx, step_deg=STEP_DEG, ray_length=RAY_LENGTH_M
    )
    userhorizon_str = ','.join(str(d) for d in horizon_degrees)
    
    print("  Computing orientation...")
    target_geom = gdf.iloc[building_idx].geometry
    orientation_results = compute_panel_orientation(target_geom)
    
    print("  Estimating peak power...")
    peakpower_kwp, area_m2 = estimate_peak_power(target_geom)
    
    # 2. Converti centroid a lat/lon
    if utm_epsg is None:
        gdf_ll = gdf.to_crs(epsg=4326)
        rep_point = gdf_ll.union_all().centroid
        lon_temp, lat_temp = rep_point.x, rep_point.y
        utm_epsg = lonlat_to_utm_epsg(lon_temp, lat_temp)
    
    centroid_ll = gpd.GeoSeries([centroid], crs=f"EPSG:{utm_epsg}").to_crs(epsg=4326).iloc[0]
    lat_pt = centroid_ll.y
    lon_pt = centroid_ll.x
    
    # 3. Chiama PVGIS (base, con horizon)
    print("  Calling PVGIS seriescalc...")
    pvgis_aspect = orientation_results['pvgis_aspect']
    
    try:
        pvgis_json = call_pvgis_seriescalc(
            lat_pt, lon_pt,
            userhorizon_str,
            peakpower=peakpower_kwp,
            aspect=pvgis_aspect
        )
        
        df_hourly = pd.DataFrame(pvgis_json['outputs']['hourly'])
    except Exception as e:
        print(f"  ERROR: PVGIS call failed: {e}")
        return None
    
    # 4. Calcola metriche base
    print("  Computing metrics...")
    annual_metrics = compute_annual_metrics(df_hourly, peakpower_kwp)
    best_worst = compute_best_worst_days(df_hourly)
    
    # Pulisci orientation per JSON (mantieni i dati necessari per plot)
    orientation_for_json = orientation_results.copy()
    orientation_for_json.pop('mbrect_geom', None)
    orientation_for_json.pop('long_sides_midpoints', None)
    
    # 5. Costruisci risultati base
    result = {
        'location': {
            'lat': lat_pt,
            'lon': lon_pt,
            'centroid_xy': (centroid.x, centroid.y),
            'crs': f"EPSG:{utm_epsg}"
        },
        'building_props': {
            'area_m2': area_m2,
            'height_m': target_height,
            'aspect_deg': orientation_results['pvgis_aspect'],
            'azimuth_deg': orientation_results['panel_azimuth_deg'],
            'peakpower_kwp': peakpower_kwp,
            'uncertain_orientation': orientation_results['uncertain_orientation_flag'],
            # Salva endpoints del lato lungo sud per plot_viewer
            'long_side_endpoints': orientation_results['chosen_long_side_endpoints'],
            'long_side_midpoint': orientation_results['chosen_long_side_midpoint']
        },
        'annual_metrics': annual_metrics,
        'best_worst_days': best_worst,
        'userhorizon_str': userhorizon_str,
        
        # On-demand (lazy load)
        'tilt_sensitivity': None,
        'horizon_impact': None,
        
        # Metadata
        '_internal': {
            'df_hourly': df_hourly,  # Mantieni per eventuali analisi future
            'pvgis_json': pvgis_json,
            'gdf': gdf
        }
    }
    
    # 6. [Opzionale] Calcola horizon impact se richiesto
    if compute_horizon_impact_flag:
        print("  Computing horizon impact...")
        result['horizon_impact'] = compute_horizon_impact(
            lat_pt, lon_pt,
            userhorizon_str,
            pvgis_aspect,
            peakpower_kwp,
            DEFAULT_TILT_FOR_ASPECT,
            annual_metrics['energy_kwh']
        )
    
    print(f"  ✓ Building {building_idx} complete")
    return result


# ============================================================
# PROCESSING TUTTI GLI EDIFICI
# ============================================================

def process_all_buildings(
    gdf,
    compute_horizon_impact_all: bool = False,
    existing_results: dict = None
) -> dict:
    """
    Loop su tutti gli edifici in GDF. Usa cache se available.
    
    Args:
        gdf: GeoDataFrame (in CRS metrico UTM)
        compute_horizon_impact_all: Se True, calcola horizon impact per tutti
        existing_results: dict di risultati già processati (per evitare riprocessare)
    
    Returns:
        dict: {0: {result}, 1: {result}, ...}
    """
    if existing_results is None:
        existing_results = {}
    
    results = existing_results.copy()
    utm_epsg = None
    
    # Calcola EPSG una volta
    gdf_ll = gdf.to_crs(epsg=4326)
    rep_point = gdf_ll.union_all().centroid
    lon_temp, lat_temp = rep_point.x, rep_point.y
    utm_epsg = lonlat_to_utm_epsg(lon_temp, lat_temp)
    
    for idx in range(len(gdf)):
        try:
            result = process_building(
                gdf, idx,
                utm_epsg=utm_epsg,
                compute_horizon_impact_flag=compute_horizon_impact_all,
                existing_results=results
            )
            if result:
                results[idx] = result
        except Exception as e:
            print(f"ERROR processing building {idx}: {e}")
            results[idx] = None
    
    return results


# ============================================================
# UPDATE ON-DEMAND (Lazy Loading)
# ============================================================

def add_tilt_sensitivity_to_building(
    results: dict,
    building_idx: int,
    gdf,
    tilt_values: list = None
) -> None:
    """
    Aggiunge analisi sensibilità tilt a un edificio specifico (in-place).
    Richiede N call aggiuntive a PVGIS.
    
    Args:
        results: Dict di risultati (da process_all_buildings)
        building_idx: Indice edificio
        gdf: GeoDataFrame (per riferimento CRS)
        tilt_values: Lista tilt da testare
    """
    if building_idx not in results or results[building_idx] is None:
        print(f"ERROR: Building {building_idx} not found in results")
        return
    
    if tilt_values is None:
        tilt_values = [15, 20, 25, 30, 35]
    
    building_data = results[building_idx]
    
    # Estrai parametri necessari
    lat = building_data['location']['lat']
    lon = building_data['location']['lon']
    userhorizon_str = building_data['userhorizon_str']
    aspect_deg = building_data['building_props']['aspect_deg']
    peakpower_kwp = building_data['building_props']['peakpower_kwp']
    
    # Calcola sensibilità
    tilt_sensitivity = compute_tilt_sensitivity(
        lat, lon,
        userhorizon_str,
        aspect_deg,
        peakpower_kwp,
        tilt_values
    )
    
    # Aggiorna in-place
    results[building_idx]['tilt_sensitivity'] = tilt_sensitivity


def add_horizon_impact_to_building(
    results: dict,
    building_idx: int,
    gdf
) -> None:
    """
    Aggiunge analisi impatto horizon a un edificio specifico (in-place).
    Richiede 1 call aggiuntiva a PVGIS.
    
    Args:
        results: Dict di risultati
        building_idx: Indice edificio
        gdf: GeoDataFrame (per riferimento)
    """
    if building_idx not in results or results[building_idx] is None:
        print(f"ERROR: Building {building_idx} not found in results")
        return
    
    building_data = results[building_idx]
    
    # Se già calcolato, skip
    if building_data['horizon_impact'] is not None:
        print(f"Building {building_idx} already has horizon_impact computed")
        return
    
    # Estrai parametri
    lat = building_data['location']['lat']
    lon = building_data['location']['lon']
    userhorizon_str = building_data['userhorizon_str']
    aspect_deg = building_data['building_props']['aspect_deg']
    peakpower_kwp = building_data['building_props']['peakpower_kwp']
    energy_with_kwh = building_data['annual_metrics']['energy_kwh']
    
    # Calcola impatto
    horizon_impact = compute_horizon_impact(
        lat, lon,
        userhorizon_str,
        aspect_deg,
        peakpower_kwp,
        DEFAULT_TILT_FOR_ASPECT,
        energy_with_kwh
    )
    
    # Aggiorna in-place
    results[building_idx]['horizon_impact'] = horizon_impact


# ============================================================
# EXPORT GEOJSON PER LEAFLET
# ============================================================

def export_geojson_for_leaflet(
    gdf,
    results: dict,
    output_path: str = 'buildings_pv_potential.geojson'
) -> None:
    """
    Esporta GeoJSON colorato per visualizzazione Leaflet.
    Colore basato su Capacity Factor.
    
    Args:
        gdf: GeoDataFrame originale
        results: Dict di risultati da process_all_buildings
        output_path: Path output GeoJSON
    """
    features = []
    
    for idx in gdf.index:
        if idx not in results or results[idx] is None:
            continue
        
        building_data = results[idx]
        geom = gdf.loc[idx, 'geometry']
        
        # Estrai metriche
        energy_kwh = building_data['annual_metrics']['energy_kwh']
        cf = building_data['annual_metrics']['capacity_factor']
        
        # Determina colore basato su CF
        if cf >= 0.20:
            color = "#2ECC71"  # Verde scuro
            category = "high"
        elif cf >= 0.15:
            color = "#F1C40F"  # Giallo
            category = "medium"
        elif cf >= 0.10:
            color = "#E67E22"  # Arancione
            category = "low"
        else:
            color = "#E74C3C"  # Rosso
            category = "very_low"
        
        # Costruisci feature
        feature = {
            "type": "Feature",
            "geometry": geom.__geo_interface__,
            "properties": {
                "building_id": int(idx),
                "energy_kwh": round(energy_kwh, 2),
                "capacity_factor": round(cf, 4),
                "cf_category": category,
                "color": color,
                "popup_text": f"Building {idx}: {energy_kwh:.0f} kWh/year, CF {cf*100:.1f}%",
                "peakpower_kwp": building_data['building_props']['peakpower_kwp']
            }
        }
        
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"\nGeoJSON exported: {output_path}")


# ============================================================
# UTILITÀ: CARICA RISULTATI DA FILE
# ============================================================

def load_pvgis_outputs(summary_json_path: str, hourly_csv_path: str) -> tuple:
    """
    Carica output PVGIS da file (legacy, per compatibilità).
    
    Args:
        summary_json_path: Path a output_summary.json
        hourly_csv_path: Path a output_hourly_data.csv
    
    Returns:
        (summary_dict, df_hourly)
    """
    with open(summary_json_path, 'r') as f:
        summary = json.load(f)
    
    df_hourly = pd.read_csv(hourly_csv_path)
    
    return summary, df_hourly


# ============================================================
# MAIN (Per testing standalone)
# ============================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print('Usage: python pvgis_analyzer.py path/to/buildings.zip [--horizon-impact]')
        sys.exit(1)
    
    zip_path = sys.argv[1]
    compute_horizon_impact_all = '--horizon-impact' in sys.argv
    
    # Extract e load
    tmpdir = tempfile.mkdtemp(prefix='pv_analysis_')
    try:
        shp_paths = extract_zip_find_shp(zip_path, tmpdir)
        shp = pick_building_shp(shp_paths)
        print(f"Using shapefile: {shp}")
        
        gdf = gpd.read_file(shp)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        
        # Reproject to UTM
        gdf_ll = gdf.to_crs(epsg=4326)
        rep_point = gdf_ll.union_all().centroid
        lon, lat = rep_point.x, rep_point.y
        utm_epsg = lonlat_to_utm_epsg(lon, lat)
        gdf = gdf.to_crs(epsg=utm_epsg)
        
        # Process all buildings
        print(f"\nProcessing {len(gdf)} buildings...\n")
        results = process_all_buildings(gdf, compute_horizon_impact_all=compute_horizon_impact_all)

        # AGGIUNGI QUESTE 2 RIGHE:
        from plot_viewer import plot_pv_potential
        plot_pv_potential(gdf, results, 'pv_potential_map.png')
        
        # Export GeoJSON
        export_geojson_for_leaflet(gdf, results)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        valid_results = [r for r in results.values() if r is not None]
        print(f"Successfully processed: {len(valid_results)}/{len(gdf)} buildings")
        
        for idx, result in results.items():
            if result:
                metrics = result['annual_metrics']
                print(f"  Building {idx}: {metrics['energy_kwh']} kWh, CF {metrics['capacity_factor']*100:.1f}%")
    
    finally:
        shutil.rmtree(tmpdir)