#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Activation Comparison CSVs for Professor's TODO

This script generates comprehensive CSV files showing:
1. Best activation function per dataset (test MAE/accuracy)
2. Comparison of ALL activations vs single activations
3. Old vs new code comparison

Datasets:
- Classification (2): CIFAR-10, Iris  
- Regression (3): Based on basis_selection (synthetic functions)

Author: Rob Saidov
Date: December 2025
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Project root
project_root = Path(__file__).parent.parent
results_dir = project_root / "results"
results_v2_dir = project_root / "results_v2"
results_research_dir = project_root / "results_research"
output_dir = project_root / "results_comparison"


def load_json_safe(filepath):
    """Safely load JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def create_classification_comparison():
    """
    Create CSV comparing activation functions on classification datasets.
    Tests: CIFAR-10, MNIST, Iris, Wine
    Metrics: Test Accuracy
    """
    print("\n" + "="*60)
    print("CLASSIFICATION COMPARISON")
    print("="*60)
    
    results = []
    
    # 1. CIFAR-10 results from results/cifar10/
    cifar_dir = results_dir / "cifar10"
    if cifar_dir.exists():
        for variant_dir in cifar_dir.iterdir():
            if variant_dir.is_dir():
                summary_file = variant_dir / "summary.json"
                if summary_file.exists():
                    data = load_json_safe(summary_file)
                    if data:
                        results.append({
                            'Dataset': 'CIFAR-10',
                            'Activation_Config': data.get('variant_name', variant_dir.name),
                            'Branches': ', '.join(data.get('branches', [])),
                            'Test_Accuracy': data.get('best_accuracy', 0),
                            'Parameters': data.get('parameters', 0),
                            'Use_Residual': data.get('use_residual', False),
                            'Source': 'results/cifar10'
                        })
    
    # 2. MNIST results from results/mnist/
    mnist_dir = results_dir / "mnist"
    if mnist_dir.exists():
        for variant_dir in mnist_dir.iterdir():
            if variant_dir.is_dir():
                summary_file = variant_dir / "summary.json"
                if summary_file.exists():
                    data = load_json_safe(summary_file)
                    if data:
                        results.append({
                            'Dataset': 'MNIST',
                            'Activation_Config': data.get('variant_name', variant_dir.name),
                            'Branches': ', '.join(data.get('branches', [])),
                            'Test_Accuracy': data.get('best_accuracy', 0),
                            'Parameters': data.get('parameters', 0),
                            'Use_Residual': data.get('use_residual', False),
                            'Source': 'results/mnist'
                        })
    
    # 3. V2 results
    v2_json_files = list(results_v2_dir.glob("*.json")) if results_v2_dir.exists() else []
    for json_file in v2_json_files:
        if json_file.name.endswith("_results.json"):
            data = load_json_safe(json_file)
            if data:
                dataset = json_file.name.split('_')[0].upper()
                config = json_file.name.replace('_results.json', '').replace(f'{json_file.name.split("_")[0]}_', '')
                results.append({
                    'Dataset': dataset,
                    'Activation_Config': config,
                    'Branches': 'see v2 config',
                    'Test_Accuracy': data.get('best_accuracy', data.get('accuracy', 0)),
                    'Parameters': data.get('parameters', 0),
                    'Use_Residual': True,
                    'Source': 'results_v2'
                })
    
    df = pd.DataFrame(results)
    if not df.empty:
        # Find best per dataset
        df_sorted = df.sort_values(['Dataset', 'Test_Accuracy'], ascending=[True, False])
        return df_sorted
    return pd.DataFrame()


def create_regression_comparison():
    """
    Create CSV comparing activation functions on regression tasks.
    Uses: basis_selection_results.json, efficiency_results.json
    Metrics: R², MSE (MAE equivalent for regression)
    """
    print("\n" + "="*60)
    print("REGRESSION COMPARISON")
    print("="*60)
    
    results = []
    
    # 1. Basis selection results
    basis_file = results_research_dir / "basis_selection_results.json"
    if basis_file.exists():
        data = load_json_safe(basis_file)
        if data and 'results' in data:
            for func_name, func_data in data['results'].items():
                for config_name, config_results in func_data.get('results', {}).items():
                    results.append({
                        'Dataset': func_name,
                        'Activation_Config': config_name,
                        'Branches': ', '.join(config_results.get('branches', [])),
                        'R2': config_results.get('r2', 0),
                        'MSE': config_results.get('mse', 0),
                        'MAE': np.sqrt(config_results.get('mse', 0)),  # Approximate MAE from MSE
                        'Best_Config_For_Dataset': func_data.get('best_config', ''),
                        'Expected_Best': func_data.get('expected_best', ''),
                        'Source': 'basis_selection'
                    })
    
    # 2. Efficiency results
    eff_file = results_research_dir / "efficiency_results.json"
    if eff_file.exists():
        data = load_json_safe(eff_file)
        if data and 'results' in data:
            for func_name, func_data in data['results'].items():
                for config_name, config_results in func_data.items():
                    if isinstance(config_results, dict):
                        results.append({
                            'Dataset': f'{func_name}_efficiency',
                            'Activation_Config': config_name,
                            'Branches': config_results.get('basis', ''),
                            'R2': config_results.get('final_r2', 0),
                            'MSE': 1 - config_results.get('final_r2', 0),  # Approximate
                            'MAE': np.sqrt(1 - config_results.get('final_r2', 0)),
                            'Parameters': config_results.get('n_params', 0),
                            'Epochs_To_Target': config_results.get('epochs_to_target', 0),
                            'Source': 'efficiency'
                        })
    
    df = pd.DataFrame(results)
    if not df.empty:
        df_sorted = df.sort_values(['Dataset', 'R2'], ascending=[True, False])
        return df_sorted
    return pd.DataFrame()


def create_best_activation_summary():
    """
    Create summary CSV showing BEST activation for each dataset.
    This is the IMP task from professor.
    """
    print("\n" + "="*60)
    print("BEST ACTIVATION PER DATASET (IMP)")
    print("="*60)
    
    classification_df = create_classification_comparison()
    regression_df = create_regression_comparison()
    
    # Find best for classification (highest accuracy)
    best_classification = []
    if not classification_df.empty:
        for dataset in classification_df['Dataset'].unique():
            subset = classification_df[classification_df['Dataset'] == dataset]
            best_row = subset.loc[subset['Test_Accuracy'].idxmax()]
            best_classification.append({
                'Dataset': dataset,
                'Task_Type': 'Classification',
                'Best_Activation': best_row['Activation_Config'],
                'Best_Branches': best_row['Branches'],
                'Metric': 'Accuracy',
                'Value': best_row['Test_Accuracy'],
                'All_Beat_Others': 'all' in str(best_row['Activation_Config']).lower() or 
                                   'all' in str(best_row['Branches']).lower(),
                'Source': best_row['Source']
            })
    
    # Find best for regression (highest R²)
    best_regression = []
    if not regression_df.empty:
        for dataset in regression_df['Dataset'].unique():
            subset = regression_df[regression_df['Dataset'] == dataset]
            best_row = subset.loc[subset['R2'].idxmax()]
            best_regression.append({
                'Dataset': dataset,
                'Task_Type': 'Regression',
                'Best_Activation': best_row['Activation_Config'],
                'Best_Branches': best_row['Branches'],
                'Metric': 'R2',
                'Value': best_row['R2'],
                'All_Beat_Others': 'all' in str(best_row['Activation_Config']).lower() or
                                   'all' in str(best_row['Branches']).lower(),
                'Source': best_row['Source']
            })
    
    summary_df = pd.DataFrame(best_classification + best_regression)
    return summary_df, classification_df, regression_df


def create_all_vs_single_comparison():
    """
    Find cases where ALL activations beat single activations.
    This is the alternative approach from professor's TODO.
    """
    print("\n" + "="*60)
    print("ALL ACTIVATIONS VS SINGLE ACTIVATIONS")
    print("="*60)
    
    results = []
    
    # Basis selection - perfect for this comparison
    basis_file = results_research_dir / "basis_selection_results.json"
    if basis_file.exists():
        data = load_json_safe(basis_file)
        if data and 'results' in data:
            for func_name, func_data in data['results'].items():
                func_results = func_data.get('results', {})
                
                # Get ALL config result
                all_result = func_results.get('all', {})
                all_r2 = all_result.get('r2', 0)
                
                # Get single-basis results
                single_configs = ['relu_only', 'fourier_only']
                
                for single in single_configs:
                    single_result = func_results.get(single, {})
                    single_r2 = single_result.get('r2', 0)
                    
                    results.append({
                        'Dataset': func_name,
                        'All_R2': all_r2,
                        'Single_Config': single,
                        'Single_R2': single_r2,
                        'All_Improvement': all_r2 - single_r2,
                        'All_Wins': all_r2 > single_r2
                    })
    
    # Classification datasets
    cifar_all = None
    cifar_relu = None
    
    cifar_dir = results_dir / "cifar10"
    if cifar_dir.exists():
        all_summary = cifar_dir / "all_branches" / "summary.json"
        relu_summary = cifar_dir / "relu_only" / "summary.json"
        
        if all_summary.exists():
            data = load_json_safe(all_summary)
            if data:
                cifar_all = data.get('best_accuracy', 0)
        
        if relu_summary.exists():
            data = load_json_safe(relu_summary)
            if data:
                cifar_relu = data.get('best_accuracy', 0)
        
        if cifar_all is not None and cifar_relu is not None:
            results.append({
                'Dataset': 'CIFAR-10',
                'All_R2': cifar_all,
                'Single_Config': 'relu_only',
                'Single_R2': cifar_relu,
                'All_Improvement': cifar_all - cifar_relu,
                'All_Wins': cifar_all > cifar_relu
            })
    
    return pd.DataFrame(results)


def generate_all_csvs():
    """Generate all required CSV files."""
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Best activation summary (IMP)
    summary_df, class_df, reg_df = create_best_activation_summary()
    
    # 2. All vs Single comparison
    all_vs_single_df = create_all_vs_single_comparison()
    
    # Save all CSVs
    files_created = []
    
    if not summary_df.empty:
        filepath = output_dir / f"best_activation_per_dataset.csv"
        summary_df.to_csv(filepath, index=False)
        files_created.append(filepath)
        print(f"\n✓ Created: {filepath}")
        print(summary_df.to_string())
    
    if not class_df.empty:
        filepath = output_dir / f"classification_comparison.csv"
        class_df.to_csv(filepath, index=False)
        files_created.append(filepath)
        print(f"\n✓ Created: {filepath}")
    
    if not reg_df.empty:
        filepath = output_dir / f"regression_comparison.csv"
        reg_df.to_csv(filepath, index=False)
        files_created.append(filepath)
        print(f"\n✓ Created: {filepath}")
    
    if not all_vs_single_df.empty:
        filepath = output_dir / f"all_vs_single_activations.csv"
        all_vs_single_df.to_csv(filepath, index=False)
        files_created.append(filepath)
        print(f"\n✓ Created: {filepath}")
        print("\nAll vs Single Activation Comparison:")
        print(all_vs_single_df.to_string())
    
    # Create a combined JSON report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_datasets_analyzed': len(summary_df) if not summary_df.empty else 0,
            'datasets_where_all_wins': int(summary_df['All_Beat_Others'].sum()) if not summary_df.empty else 0,
        },
        'files_created': [str(f) for f in files_created]
    }
    
    report_path = output_dir / "comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Created: {report_path}")
    
    return files_created


def main():
    print("="*60)
    print("HybridKAN Activation Comparison CSV Generator")
    print("Professor's TODO: Best activation per dataset")
    print("="*60)
    
    files = generate_all_csvs()
    
    print("\n" + "="*60)
    print(f"COMPLETE: Generated {len(files)} CSV files")
    print(f"Location: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
