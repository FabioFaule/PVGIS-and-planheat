#!/usr/bin/env python
"""
test_tilt_sensitivity.py

Testa l'analisi di sensibilità del tilt per UN building specifico.
Genera grafici per visualizzare come varia l'energia al variare dell'angolo.

Uso:
    python test_tilt_sensitivity.py shapetest.zip --building-idx 20
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pvgis_horizon_from_shapefile import (
    extract_zip_find_shp,
    pick_building_shp,
    lonlat_to_utm_epsg,
    compute_userhorizon_from_gdf,
    compute_panel_orientation,
    estimate_peak_power,
    estimate_height_attr,
    call_pvgis_seriescalc,
)

from pvgis_analyzer import (
    compute_annual_metrics,
    compute_best_worst_days,
    compute_tilt_sensitivity,
    STEP_DEG,
    RAY_LENGTH_M,
)


def test_single_building_tilt(zip_path, building_idx=20, tilt_values=None):
    """
    Processa UN building e fa sensitivity analysis completo.
    
    Args:
        zip_path: Path al ZIP con shapefile
        building_idx: Indice building da analizzare
        tilt_values: Lista tilt da testare (default: [10, 15, 20, 25, 30, 35, 40])
    """
    
    if tilt_values is None:
        tilt_values = [10, 15, 20, 25, 30, 35, 40]
    
    tmpdir = tempfile.mkdtemp(prefix='pv_test_')
    
    try:
        print(f"\n{'='*70}")
        print(f"TILT SENSITIVITY TEST - Building {building_idx}")
        print(f"{'='*70}\n")
        
        # Extract e load
        print("[1/6] Extracting shapefile...")
        shp_paths = extract_zip_find_shp(zip_path, tmpdir)
        shp = pick_building_shp(shp_paths)
        
        gdf = gpd.read_file(shp)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        
        print(f"  ✓ Loaded {len(gdf)} buildings")
        
        # Reproject to UTM
        print("\n[2/6] Reprojecting to UTM...")
        gdf_ll = gdf.to_crs(epsg=4326)
        rep_point = gdf_ll.union_all().centroid
        lon, lat = rep_point.x, rep_point.y
        utm_epsg = lonlat_to_utm_epsg(lon, lat)
        gdf = gdf.to_crs(epsg=utm_epsg)
        print(f"  ✓ Using EPSG:{utm_epsg}")
        
        # Check building index
        if building_idx < 0 or building_idx >= len(gdf):
            raise IndexError(f"Building index {building_idx} out of range [0, {len(gdf)-1}]")
        
        # Compute horizon
        print(f"\n[3/6] Computing horizon for building {building_idx}...")
        horizon_items, horizon_degrees, centroid, target_height = compute_userhorizon_from_gdf(
            gdf, target_idx=building_idx, step_deg=STEP_DEG, ray_length=RAY_LENGTH_M
        )
        userhorizon_str = ','.join(str(d) for d in horizon_degrees)
        print(f"  ✓ Horizon computed ({len(horizon_degrees)} angles)")
        
        # Compute orientation
        print(f"\n[4/6] Computing orientation...")
        target_geom = gdf.iloc[building_idx].geometry
        orientation_results = compute_panel_orientation(target_geom)
        peakpower_kwp, area_m2 = estimate_peak_power(target_geom)
        
        print(f"  ✓ Azimuth: {orientation_results['panel_azimuth_deg']}°")
        print(f"  ✓ Aspect: {orientation_results['pvgis_aspect']}°")
        print(f"  ✓ Peak Power: {peakpower_kwp} kWp")
        
        # Convert centroid to lat/lon
        centroid_ll = gpd.GeoSeries([centroid], crs=f"EPSG:{utm_epsg}").to_crs(epsg=4326).iloc[0]
        lat_pt = centroid_ll.y
        lon_pt = centroid_ll.x
        
        pvgis_aspect = orientation_results['pvgis_aspect']
        
        # Compute base scenario (tilt=35, con horizon)
        print(f"\n[5/6] Computing base scenario (tilt=35° with horizon)...")
        try:
            pvgis_base = call_pvgis_seriescalc(
                lat_pt, lon_pt,
                userhorizon_str,
                peakpower_kwp,
                tilt=35,
                aspect=pvgis_aspect
            )
            df_base = pd.DataFrame(pvgis_base['outputs']['hourly'])
            base_metrics = compute_annual_metrics(df_base, peakpower_kwp)
            base_energy = base_metrics['energy_kwh']
            best_worst = compute_best_worst_days(df_base)
            
            print(f"  ✓ Base Energy (tilt=35°): {base_energy:.2f} kWh/year")
            print(f"    - Capacity Factor: {base_metrics['capacity_factor']*100:.2f}%")
            print(f"    - Peak Hours: {base_metrics['peak_hours_h']:.2f} h")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            return None
        
        # Sensitivity analysis
        print(f"\n[6/6] Computing tilt sensitivity ({len(tilt_values)} scenarios)...")
        tilt_sensitivity = compute_tilt_sensitivity(
            lat_pt, lon_pt,
            userhorizon_str,
            pvgis_aspect,
            peakpower_kwp,
            tilt_values
        )
        
        # Summary
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}\n")
        
        print("Tilt Sensitivity Analysis:")
        for tilt, energy in zip(tilt_sensitivity['tilt_values'], tilt_sensitivity['energy_values']):
            gain_loss = energy - base_energy
            gain_loss_pct = (gain_loss / base_energy * 100) if base_energy > 0 else 0
            marker = "← OPTIMAL" if tilt == tilt_sensitivity['optimal_tilt'] else ""
            print(f"  {tilt:2d}° : {energy:8.0f} kWh  ({gain_loss:+7.0f} kWh, {gain_loss_pct:+5.1f}%) {marker}")
        
        print(f"\n  ► Optimal Tilt: {tilt_sensitivity['optimal_tilt']}°")
        print(f"    Energy at optimal: {tilt_sensitivity['optimal_energy_kwh']:.2f} kWh/year")
        gain_at_optimal = tilt_sensitivity['optimal_energy_kwh'] - base_energy
        gain_pct = (gain_at_optimal / base_energy * 100) if base_energy > 0 else 0
        print(f"    Gain vs tilt 35°: {gain_at_optimal:+.2f} kWh ({gain_pct:+.2f}%)")
        
        print(f"\nBest/Worst Days (tilt=35°):")
        print(f"  Best:  {best_worst['best']['date']} → {best_worst['best']['energy_kwh']:.2f} kWh")
        print(f"  Worst: {best_worst['worst']['date']} → {best_worst['worst']['energy_kwh']:.2f} kWh")
        
        # Plot 1: Matplotlib static
        print(f"\n{'='*70}")
        print("Generating plots...")
        print(f"{'='*70}\n")
        
        plot_tilt_sensitivity_matplotlib(tilt_sensitivity, base_energy, 'tilt_sensitivity_static.png')
        
        # Plot 2: Plotly interactive
        plot_tilt_sensitivity_plotly(tilt_sensitivity, base_energy, 'tilt_sensitivity_interactive.html')
        
        # Plot 3: Best/Worst day profiles
        plot_best_worst_profiles(best_worst, 'best_worst_profiles.png')
        
        print(f"\n{'='*70}")
        print("✓ Test Complete!")
        print(f"{'='*70}\n")
        
        return {
            'building_idx': building_idx,
            'base_energy': base_energy,
            'tilt_sensitivity': tilt_sensitivity,
            'best_worst': best_worst,
            'metrics': base_metrics,
            'orientation': orientation_results
        }
        
    finally:
        shutil.rmtree(tmpdir)


def plot_tilt_sensitivity_matplotlib(tilt_sensitivity, base_energy, output_path):
    """Plot statico matplotlib per sensibilità tilt."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    tilts = tilt_sensitivity['tilt_values']
    energies = tilt_sensitivity['energy_values']
    optimal_tilt = tilt_sensitivity['optimal_tilt']
    optimal_energy = tilt_sensitivity['optimal_energy_kwh']
    
    # Plot linea
    ax.plot(tilts, energies, 'b-o', linewidth=2, markersize=8, label='Energy')
    
    # Highlight ottimale
    ax.plot(optimal_tilt, optimal_energy, 'r*', markersize=20, label=f'Optimal ({optimal_tilt}°)')
    
    # Highlight base (35°)
    if 35 in tilts:
        base_idx = tilts.index(35)
        ax.plot(35, energies[base_idx], 'g^', markersize=12, label='Base (35°)')
    
    # Aggiungi etichette su ogni punto
    for tilt, energy in zip(tilts, energies):
        ax.annotate(f'{energy:.0f}', 
                   xy=(tilt, energy), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9)
    
    ax.set_xlabel('Tilt Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Energy (kWh)', fontsize=12, fontweight='bold')
    ax.set_title('Tilt Sensitivity Analysis - Annual Energy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Static plot: {output_path}")
    plt.close(fig)


def plot_tilt_sensitivity_plotly(tilt_sensitivity, base_energy, output_path):
    """Plot interattivo Plotly per sensibilità tilt."""
    
    tilts = tilt_sensitivity['tilt_values']
    energies = tilt_sensitivity['energy_values']
    optimal_tilt = tilt_sensitivity['optimal_tilt']
    optimal_energy = tilt_sensitivity['optimal_energy_kwh']
    
    # Calcola gain/loss rispetto a base
    gains = [e - base_energy for e in energies]
    gains_pct = [(g / base_energy * 100) if base_energy > 0 else 0 for g in gains]
    
    # Hover text
    hover_text = [
        f"Tilt: {tilt}°<br>" +
        f"Energy: {energy:.0f} kWh<br>" +
        f"Gain: {gain:+.0f} kWh ({gain_pct:+.1f}%)"
        for tilt, energy, gain, gain_pct in zip(tilts, energies, gains, gains_pct)
    ]
    
    fig = go.Figure()
    
    # Main line
    fig.add_trace(go.Scatter(
        x=tilts, y=energies,
        mode='lines+markers',
        name='Annual Energy',
        line=dict(color='blue', width=3),
        marker=dict(size=10),
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    # Optimal point
    fig.add_trace(go.Scatter(
        x=[optimal_tilt], y=[optimal_energy],
        mode='markers',
        name=f'Optimal ({optimal_tilt}°)',
        marker=dict(color='red', size=15, symbol='star'),
        hovertext=[f"Optimal: {optimal_tilt}° → {optimal_energy:.0f} kWh"],
        hoverinfo='text'
    ))
    
    # Base point (35°)
    if 35 in tilts:
        base_idx = tilts.index(35)
        fig.add_trace(go.Scatter(
            x=[35], y=[energies[base_idx]],
            mode='markers',
            name=f'Base (35°)',
            marker=dict(color='green', size=12, symbol='triangle-up'),
            hovertext=[f"Base: 35° → {energies[base_idx]:.0f} kWh"],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title='Tilt Sensitivity Analysis - Interactive',
        xaxis_title='Tilt Angle (degrees)',
        yaxis_title='Annual Energy (kWh)',
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        font=dict(size=12),
        height=600,
        width=1000,
    )
    
    fig.write_html(output_path)
    print(f"  ✓ Interactive plot: {output_path}")


def plot_best_worst_profiles(best_worst, output_path):
    """Plot profili orari best vs worst day."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    hours = list(range(24))
    best_profile = best_worst['best']['profile']
    worst_profile = best_worst['worst']['profile']
    best_date = best_worst['best']['date']
    worst_date = best_worst['worst']['date']
    best_energy = best_worst['best']['energy_kwh']
    worst_energy = best_worst['worst']['energy_kwh']
    
    # Plot Best Day
    ax1.fill_between(hours, best_profile, alpha=0.3, color='green')
    ax1.plot(hours, best_profile, 'g-o', linewidth=2, markersize=6)
    ax1.set_title(f'Best Day: {best_date} ({best_energy:.2f} kWh)', 
                  fontsize=13, fontweight='bold', color='green')
    ax1.set_ylabel('Power (W)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_xticks(range(0, 24, 2))
    
    # Plot Worst Day
    ax2.fill_between(hours, worst_profile, alpha=0.3, color='red')
    ax2.plot(hours, worst_profile, 'r-o', linewidth=2, markersize=6)
    ax2.set_title(f'Worst Day: {worst_date} ({worst_energy:.2f} kWh)', 
                  fontsize=13, fontweight='bold', color='red')
    ax2.set_xlabel('Hour of Day', fontsize=11)
    ax2.set_ylabel('Power (W)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, 23.5)
    ax2.set_xticks(range(0, 24, 2))
    
    fig.suptitle('Best vs Worst Day Hourly Profiles', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Best/Worst profiles: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('Usage: python test_tilt_sensitivity.py path/to/buildings.zip [--building-idx 20]')
        print('\nExample:')
        print('  python test_tilt_sensitivity.py shapetest.zip')
        print('  python test_tilt_sensitivity.py shapetest.zip --building-idx 20')
        sys.exit(1)
    
    zip_path = sys.argv[1]
    
    # Parse building index
    building_idx = 20
    if '--building-idx' in sys.argv:
        idx_arg = sys.argv.index('--building-idx')
        if idx_arg + 1 < len(sys.argv):
            building_idx = int(sys.argv[idx_arg + 1])
    
    if not os.path.isfile(zip_path):
        print(f'ZIP file not found: {zip_path}')
        sys.exit(1)
    
    # Run test
    result = test_single_building_tilt(zip_path, building_idx=building_idx)