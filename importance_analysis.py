# -*- coding: utf-8 -*-
# ==============================================================================
# M5 — Model Importance Analysis
# - Analyze feature importance for 10 store models
# - Generate importance charts and summary reports
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import xgboost as xgb

# ----------------- Configuration -----------------
OUT_DIR = "./ouput/feature_by_store"
RESULTS_DIR = "./ouput/importance_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set font for plots
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ----------------- Importance Analysis Functions -----------------
def analyze_model_importance(model_path, store_name):
    """
    Analyze feature importance for a single model
    """
    try:
        # Load model
        model = xgb.Booster()
        model.load_model(model_path)
        
        # Get importance scores
        importance_gain = model.get_score(importance_type='gain')
        importance_weight = model.get_score(importance_type='weight')
        importance_cover = model.get_score(importance_type='cover')
        
        # Convert to DataFrame
        importance_data = []
        all_features = set(importance_gain.keys()) | set(importance_weight.keys()) | set(importance_cover.keys())
        
        for feature in all_features:
            importance_data.append({
                'feature': feature,
                'gain': importance_gain.get(feature, 0),
                'weight': importance_weight.get(feature, 0),
                'cover': importance_cover.get(feature, 0),
                'store': store_name
            })
        
        importance_df = pd.DataFrame(importance_data)
        
        # Normalize importance scores (0-1 range)
        for col in ['gain', 'weight', 'cover']:
            if len(importance_df) > 0 and importance_df[col].sum() > 0:
                importance_df[f'{col}_normalized'] = importance_df[col] / importance_df[col].sum()
            else:
                importance_df[f'{col}_normalized'] = 0
        
        return importance_df, model
        
    except Exception as e:
        print(f"Error analyzing model {store_name}: {e}")
        return None, None

def plot_importance_comparison(all_importance_df, importance_type='gain', top_n=15):
    """
    Plot feature importance comparison across all stores
    """
    # Calculate average importance
    avg_importance = (all_importance_df.groupby('feature')[f'{importance_type}_normalized']
                     .mean()
                     .sort_values(ascending=False)
                     .head(top_n))
    
    top_features = avg_importance.index.tolist()
    
    # Filter data
    plot_data = all_importance_df[all_importance_df['feature'].isin(top_features)]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Subplot 1: Heatmap of importance across stores
    pivot_data = plot_data.pivot_table(
        index='store', 
        columns='feature', 
        values=f'{importance_type}_normalized',
        fill_value=0
    )
    
    # Sort columns by average importance
    pivot_data = pivot_data[top_features]
    
    sns.heatmap(
        pivot_data, 
        ax=ax1, 
        cmap='YlOrRd', 
        annot=True, 
        fmt='.3f',
        cbar_kws={'label': f'{importance_type.title()} Importance'}
    )
    ax1.set_title(f'Feature Importance Comparison Across Stores - {importance_type.upper()}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Stores')
    
    # Subplot 2: Average importance bar chart
    colors = plt.cm.YlOrRd(np.linspace(0.6, 1, len(avg_importance)))
    bars = ax2.barh(range(len(avg_importance)), avg_importance.values, color=colors)
    ax2.set_yticks(range(len(avg_importance)))
    ax2.set_yticklabels(avg_importance.index)
    ax2.set_xlabel(f'Average Normalized {importance_type.title()} Importance')
    ax2.set_title(f'Top {top_n} Features Average Importance - {importance_type.upper()}', 
                  fontsize=14, fontweight='bold')
    
    # Add values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'importance_comparison_{importance_type}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return top_features

def plot_individual_store_importance(importance_df, store_name, importance_type='gain', top_n=15):
    """
    Plot feature importance for a single store
    """
    if importance_df.empty:
        return
    
    # Sort and select top features
    top_features = (importance_df.nlargest(top_n, f'{importance_type}_normalized')
                    .sort_values(f'{importance_type}_normalized'))
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(top_features)), 
                    top_features[f'{importance_type}_normalized'],
                    color=plt.cm.YlOrRd(np.linspace(0.6, 1, len(top_features))))
    
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel(f'Normalized {importance_type.title()} Importance')
    plt.title(f'{store_name} - Top {top_n} Feature Importance ({importance_type.upper()})', 
              fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{store_name}_importance_{importance_type}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_importance_report(all_importance_df):
    """
    Create importance analysis report
    """
    report = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_stores': all_importance_df['store'].nunique(),
        'total_features': all_importance_df['feature'].nunique(),
        'stores_analyzed': all_importance_df['store'].unique().tolist()
    }
    
    # Analyze by importance type
    importance_types = ['gain', 'weight', 'cover']
    
    for imp_type in importance_types:
        # Calculate average importance for each feature
        avg_importance = (all_importance_df.groupby('feature')[f'{imp_type}_normalized']
                         .mean()
                         .sort_values(ascending=False))
        
        top_10_features = avg_importance.head(10)
        
        report[f'top_10_features_{imp_type}'] = {
            'features': top_10_features.index.tolist(),
            'scores': top_10_features.values.tolist()
        }
        
        # Calculate importance stability (standard deviation across stores)
        importance_stability = (all_importance_df.groupby('feature')[f'{imp_type}_normalized']
                              .std()
                              .sort_values())
        
        report[f'most_stable_features_{imp_type}'] = importance_stability.head(10).index.tolist()
        report[f'most_variable_features_{imp_type}'] = importance_stability.tail(10).index.tolist()
    
    return report

# ----------------- Main Function -----------------
def main():
    print("========== M5 Model Importance Analysis ==========")
    print("Starting feature importance analysis...")
    
    # Find all model files
    model_files = []
    for file in os.listdir(OUT_DIR):
        if file.startswith("model_") and file.endswith("_xgb.json"):
            store_name = file.replace("model_", "").replace("_xgb.json", "")
            model_files.append((store_name, os.path.join(OUT_DIR, file)))
    
    print(f"Found {len(model_files)} model files")
    
    all_importance_dfs = []
    
    # Analyze each model
    for store_name, model_path in model_files:
        print(f"Analyzing {store_name}...")
        
        importance_df, model = analyze_model_importance(model_path, store_name)
        
        if importance_df is not None and not importance_df.empty:
            all_importance_dfs.append(importance_df)
            
            # Generate individual importance plots for each store
            for imp_type in ['gain', 'weight', 'cover']:
                plot_individual_store_importance(importance_df, store_name, imp_type)
            
            print(f"  {store_name} completed - {len(importance_df)} features")
        else:
            print(f"  {store_name} analysis failed")
    
    if not all_importance_dfs:
        print("No model data found for analysis")
        return
    
    # Combine all importance data
    combined_importance_df = pd.concat(all_importance_dfs, ignore_index=True)
    
    # Generate comparison plots
    top_features_by_type = {}
    for imp_type in ['gain', 'weight', 'cover']:
        top_features = plot_importance_comparison(combined_importance_df, imp_type)
        top_features_by_type[imp_type] = top_features
    
    # Create analysis report
    report = create_importance_report(combined_importance_df)
    
    # Save report
    with open(os.path.join(RESULTS_DIR, 'importance_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save detailed data
    combined_importance_df.to_csv(os.path.join(RESULTS_DIR, 'all_importance_data.csv'), index=False)
    
    # Generate summary table
    summary_table = (combined_importance_df.groupby('feature')[['gain_normalized', 'weight_normalized', 'cover_normalized']]
                    .mean()
                    .sort_values('gain_normalized', ascending=False))
    
    summary_table.to_csv(os.path.join(RESULTS_DIR, 'feature_importance_summary.csv'))
    
    # Print key findings
    print("\n" + "="*50)
    print("Importance analysis completed!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Stores analyzed: {report['total_stores']}")
    print(f"Total features: {report['total_features']}")
    print("\nTop 5 Features (GAIN):")
    for i, (feature, score) in enumerate(zip(report['top_10_features_gain']['features'][:5], 
                                           report['top_10_features_gain']['scores'][:5])):
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    print("\nGenerated files:")
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(('.png', '.csv', '.json')):
            print(f"  - {file}")

if __name__ == "__main__":
    main()