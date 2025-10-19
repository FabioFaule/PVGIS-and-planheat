#!/usr/bin/env python
"""
plot_viewer.py

Visualizza il potenziale fotovoltaico di tutti gli edifici:
- Edifici in grigio
- Rettangoli pannelli SUD colorati per energia (quintili)
- Legenda numerica
- Coordinate UTM

Uso:
    python plot_viewer.py results_dict gdf output.png
    
Oppure:
    from plot_viewer import plot_pv_potential
    plot_pv_potential(gdf, results, 'output.png')
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point


def create_panel_rectangle(centroid, long_side_p1, long_side_p2):
    """
    Crea rettangolo pannelli usando 3 step:
    1. Lato lungo: p1 → p2
    2. Perpendicolare dal centroide al lato lungo
    3. Rettangolo: base = lato lungo, altezza = distanza perpendicolare
    
    Args:
        centroid: Point (shapely)
        long_side_p1, long_side_p2: tuple (x, y) - endpoints del lato lungo
    
    Returns:
        Polygon rettangolo pannelli
    """
    
    p1 = np.array(long_side_p1)
    p2 = np.array(long_side_p2)
    c = np.array([centroid.x, centroid.y])
    
    # Step 1: Vettore lato lungo
    lato_vec = p2 - p1
    lato_len = np.linalg.norm(lato_vec)
    
    if lato_len < 1e-6:
        return None
    
    # Step 2: Vettore perpendicolare al lato (ruota 90° verso il centroide)
    # Perpendicolare: (-y, x) oppure (y, -x)
    perp_vec = np.array([-lato_vec[1], lato_vec[0]])
    perp_vec = perp_vec / np.linalg.norm(perp_vec)
    
    # Determina direzione perpendicolare verso centroide
    to_centroid = c - (p1 + p2) / 2  # vettore dal midpoint verso centroide
    if np.dot(perp_vec, to_centroid) < 0:
        perp_vec = -perp_vec
    
    # Step 3: Calcola proiezione perpendiculare del centroide sul lato
    # Proiezione: t = (c - p1) · (p2 - p1) / |p2 - p1|²
    t = np.dot(c - p1, lato_vec) / (lato_len ** 2)
    t = np.clip(t, 0, 1)  # Limita tra p1 e p2
    
    proj_point = p1 + t * lato_vec
    
    # Distanza perpendicolare
    perp_dist = np.linalg.norm(c - proj_point)
    
    if perp_dist < 1e-6:
        # Centroide già sul lato, usa piccola altezza
        perp_dist = lato_len * 0.1
    
    # Costruisci rettangolo: 4 vertici
    # Base = lato lungo (p1, p2)
    # Altezza = distanza perpendicolare (dal lato verso centroide)
    
    p3 = p2 + perp_dist * perp_vec
    p4 = p1 + perp_dist * perp_vec
    
    rect_polygon = Polygon([p1, p2, p3, p4])
    
    return rect_polygon


def compute_quintiles(values):
    """
    Calcola quintili (0%, 20%, 40%, 60%, 80%, 100%) per distribuire i colori.
    
    Args:
        values: list di valori numerici
    
    Returns:
        (quintile_boundaries, quintile_labels)
    """
    values_sorted = sorted([v for v in values if v is not None and v > 0])
    
    if len(values_sorted) == 0:
        return [0, 1], ["0-1"]
    
    # Calcola quintili
    q0 = min(values_sorted)
    q20 = np.percentile(values_sorted, 20)
    q40 = np.percentile(values_sorted, 40)
    q60 = np.percentile(values_sorted, 60)
    q80 = np.percentile(values_sorted, 80)
    q100 = max(values_sorted)
    
    boundaries = [q0, q20, q40, q60, q80, q100]
    
    labels = [
        f"{q0:.0f}-{q20:.0f}",
        f"{q20:.0f}-{q40:.0f}",
        f"{q40:.0f}-{q60:.0f}",
        f"{q60:.0f}-{q80:.0f}",
        f"{q80:.0f}-{q100:.0f}"
    ]
    
    return boundaries, labels


def value_to_quintile(value, boundaries):
    """
    Assegna un valore al suo quintile (0-4).
    
    Args:
        value: valore numerico
        boundaries: lista 6 elementi da compute_quintiles
    
    Returns:
        int (0-4) - indice quintile
    """
    if value is None or value <= 0:
        return 0
    
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= value < boundaries[i + 1]:
            return i
    
    return len(boundaries) - 2


def get_quintile_colors():
    """
    Ritorna 5 colori per i quintili (rosso → verde).
    
    Returns:
        list di 5 colori hex
    """
    # Palette: rosso → giallo → verde
    colors = [
        '#d73027',  # Rosso scuro (Q1 - basso)
        '#fc8d59',  # Arancione (Q2)
        '#fee08b',  # Giallo (Q3 - medio)
        '#91bfdb',  # Azzurro (Q4)
        '#1a9850'   # Verde scuro (Q5 - alto)
    ]
    return colors


def plot_pv_potential(gdf, results, output_path='pv_potential_map.png', figsize=(20, 20), dpi=150):
    """
    Plotta il potenziale fotovoltaico di tutti gli edifici.
    
    Args:
        gdf: GeoDataFrame degli edifici (in CRS metrico UTM)
        results: dict di risultati da process_all_buildings()
        output_path: Path output PNG
        figsize: Dimensioni figura (default 20x20)
        dpi: Risoluzione (default 150)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Estrai energie per quintili
    energy_values = []
    for idx, result in results.items():
        if result and 'annual_metrics' in result:
            energy_values.append(result['annual_metrics']['energy_kwh'])
        else:
            energy_values.append(0)
    
    boundaries, q_labels = compute_quintiles(energy_values)
    colors = get_quintile_colors()
    
    # Plot background: tutti gli edifici in grigio
    gdf.plot(ax=ax, color='lightgrey', edgecolor='darkgrey', alpha=0.6, linewidth=0.5)
    
    # Plot pannelli colorati per quintile
    for idx, row in gdf.iterrows():
        if idx not in results or results[idx] is None:
            continue
        
        result = results[idx]
        building_data = result
        
        # Estrai geometria pannelli
        try:
            # Estrai endpoints del lato lungo sud da building_props (salvati in pvgis_analyzer)
            building_props = building_data.get('building_props', {})
            long_side_endpoints = building_props.get('long_side_endpoints')
            
            if not long_side_endpoints or len(long_side_endpoints) != 2:
                continue
            
            p1, p2 = long_side_endpoints
            geom = row.geometry
            centroid = geom.centroid
            
            # Crea rettangolo pannelli usando i 3 step
            panel_rect = create_panel_rectangle(centroid, p1, p2)
            
            if panel_rect is None or panel_rect.is_empty:
                continue
            
            # Determina colore per quintile
            energy = building_data['annual_metrics']['energy_kwh']
            quintile_idx = value_to_quintile(energy, boundaries)
            color = colors[quintile_idx]
            
            # Plot rettangolo
            patch = patches.Polygon(
                list(panel_rect.exterior.coords[:-1]),
                facecolor=color,
                edgecolor='black',
                linewidth=0.3,
                alpha=0.8,
                zorder=5
            )
            ax.add_patch(patch)
            
        except Exception as e:
            # Se fallisce per un edificio, continua con gli altri
            print(f"  Warning: Could not plot panels for building {idx}: {e}")
            continue
    
    # Configura assi
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('PV Potential Map - Annual Energy (kWh)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legenda con quintili
    legend_elements = []
    for i, (label, color) in enumerate(zip(q_labels, colors)):
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', label=label,
                   markerfacecolor=color, markeredgecolor='black', markersize=10)
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
              title='Energy (kWh/year)', title_fontsize=12, framealpha=0.95)
    
    # Tight layout
    plt.tight_layout()
    
    # Salva
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\n✓ PV Potential map saved: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    import sys
    import json
    from pathlib import Path
    
    # Uso standalone: python plot_viewer.py results_dict.json gdf.pkl output.png
    if len(sys.argv) < 3:
        print('Usage: python plot_viewer.py <results_json> <gdf_geojson> <output_png>')
        print('  or import and use: plot_pv_potential(gdf, results, output_path)')
        sys.exit(1)
    
    results_path = sys.argv[1]
    gdf_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'pv_potential_map.png'
    
    # Carica results da JSON
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Carica GeoDataFrame da GeoJSON
    gdf = gpd.read_file(gdf_path)
    
    # Plot
    plot_pv_potential(gdf, results, output_path)