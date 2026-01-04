import os
import re
from pathlib import Path
import csv
from collections import defaultdict

def extract_test_results(log_path):
    """Extract test accuracy and macro_f1 from log file"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Find the last occurrence of test results
        # Updated pattern to handle numbers with commas (e.g., "2,465")
        pattern = r"Evaluate on the \*test\* set\n=> result\n\* total: ([\d,]+)\n\* correct: ([\d,]+)\n\* accuracy: ([\d.]+)%\n\* error: ([\d.]+)%\n\* macro_f1: ([\d.]+)%"
        matches = list(re.finditer(pattern, content))
        
        if matches:
            last_match = matches[-1]
            # Remove commas from numbers
            total = last_match.group(1).replace(',', '')
            correct = last_match.group(2).replace(',', '')
            accuracy = last_match.group(3)
            error = last_match.group(4)
            macro_f1 = last_match.group(5)
            
            return {
                'total': total,
                'correct': correct,
                'accuracy': float(accuracy),
                'error': float(error),
                'macro_f1': float(macro_f1)
            }
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    
    return None

def parse_path_info(log_path, output_dir):
    """Parse dataset, noise type, noise rate, and seed from path"""
    rel_path = os.path.relpath(log_path, output_dir)
    parts = Path(rel_path).parts
    
    # Expected structure: caltech101/NLPrompt/rn50_16shots/noise_asym_0.500/seed1/log.txt
    if len(parts) >= 6:
        dataset = parts[0]
        model = parts[1]  # NLPrompt
        shots = parts[2]  # rn50_16shots
        noise_config = parts[3]  # noise_asym_0.500
        seed = parts[4]  # seed1
        
        # Parse noise type and rate from noise_config
        # e.g., "noise_asym_0.500" or "noise_sym_0.0"
        noise_parts = noise_config.split('_')
        if len(noise_parts) == 3 and noise_parts[0] == 'noise':
            noise_type = noise_parts[1]  # 'asym' or 'sym'
            try:
                noise_rate = float(noise_parts[2])
            except ValueError:
                noise_rate = 0.0
        else:
            noise_type = 'unknown'
            noise_rate = 0.0
        
        return {
            'dataset': dataset,
            'model': model,
            'shots': shots,
            'noise_type': noise_type,
            'noise_rate': noise_rate,
            'seed': seed
        }
    
    return None

def collect_all_results(output_dir):
    """Collect all results from output directory"""
    results = []
    
    for root, dirs, files in os.walk(output_dir):
        if 'log.txt' in files:
            log_path = os.path.join(root, 'log.txt')
            
            # Extract test results
            test_results = extract_test_results(log_path)
            if test_results is None:
                print(f"Warning: Could not extract results from {log_path}")
                continue
            
            # Parse path information
            path_info = parse_path_info(log_path, output_dir)
            if path_info is None:
                print(f"Warning: Could not parse path info from {log_path}")
                continue
            
            # Combine information
            result = {**path_info, **test_results}
            results.append(result)
            
            print(f"✓ Processed: {path_info['dataset']:15s} | {path_info['noise_type']:5s} | {path_info['noise_rate']:.3f} | {path_info['seed']:5s} | Acc: {test_results['accuracy']:.2f}%")
    
    return results

def save_to_csv(results, output_file):
    """Save results to CSV file"""
    if not results:
        print("No results to save!")
        return
    
    fieldnames = ['dataset', 'model', 'shots', 'noise_type', 'noise_rate', 'seed', 
                  'accuracy', 'macro_f1', 'error', 'total', 'correct']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort results by dataset, noise_type, noise_rate, seed
        sorted_results = sorted(results, key=lambda x: (
            x['dataset'], 
            x['noise_type'], 
            x['noise_rate'], 
            x['seed']
        ))
        
        for result in sorted_results:
            writer.writerow(result)
    
    print(f"\n✓ Results saved to {output_file}")

def save_to_markdown(results, output_file):
    """Save results to Markdown file with summary tables"""
    if not results:
        print("No results to save!")
        return
    
    # Group results by dataset
    dataset_results = defaultdict(list)
    for result in results:
        dataset_results[result['dataset']].append(result)
    
    with open(output_file, 'w') as f:
        f.write("# NLPrompt Experimental Results\n\n")
        f.write(f"**Total Experiments:** {len(results)}\n\n")
        f.write("---\n\n")
        
        for dataset in sorted(dataset_results.keys()):
            f.write(f"## {dataset.upper()}\n\n")
            
            # Group by noise type
            noise_types = defaultdict(list)
            for result in dataset_results[dataset]:
                noise_types[result['noise_type']].append(result)
            
            for noise_type in sorted(noise_types.keys()):
                f.write(f"### {noise_type.capitalize()} Noise\n\n")
                
                # Create detailed table
                f.write("| Noise Rate | Seed | Accuracy (%) | Macro F1 (%) | Total | Correct |\n")
                f.write("|------------|------|--------------|--------------|-------|----------|\n")
                
                # Sort by noise rate and seed
                sorted_noise = sorted(noise_types[noise_type], 
                                    key=lambda x: (x['noise_rate'], x['seed']))
                
                for result in sorted_noise:
                    f.write(f"| {result['noise_rate']:.3f} | {result['seed']} | "
                          f"{result['accuracy']:.2f} | {result['macro_f1']:.2f} | "
                          f"{result['total']} | {result['correct']} |\n")
                
                
        
        f.write("---\n\n")
        f.write("*Report generated automatically*\n")
    
    print(f"✓ Markdown report saved to {output_file}")

def print_summary(results):
    """Print a summary of collected results"""
    if not results:
        return
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    datasets = defaultdict(lambda: defaultdict(set))
    for result in results:
        datasets[result['dataset']][result['noise_type']].add(result['noise_rate'])
    
    for dataset in sorted(datasets.keys()):
        print(f"\n{dataset.upper()}:")
        for noise_type in sorted(datasets[dataset].keys()):
            rates = sorted(datasets[dataset][noise_type])
            print(f"  {noise_type.capitalize()} noise: {len(rates)} rate(s) - {rates}")
    
    print(f"\nTotal experiments: {len(results)}")
    print("="*80)

def main():
    # Set paths
    output_dir = "/home/wangzh/code-sapce/NLPrompt/output"
    csv_file = "/home/wangzh/code-sapce/NLPrompt/results_summary.csv"
    md_file = "/home/wangzh/code-sapce/NLPrompt/results_summary.md"
    
    print("="*80)
    print("NLPrompt Results Extraction")
    print("="*80)
    print(f"Output directory: {output_dir}\n")
    
    print("Collecting results from log files...\n")
    results = collect_all_results(output_dir)
    
    if results:
        # Print summary
        print_summary(results)
        
        # Save to CSV
        save_to_csv(results, csv_file)
        
        # Save to Markdown
        save_to_markdown(results, md_file)
        
        print("\n" + "="*80)
        print("✓ Extraction completed successfully!")
        print("="*80)
        print(f"CSV file: {csv_file}")
        print(f"Markdown file: {md_file}")
    else:
        print("\n" + "="*80)
        print("✗ No results found!")
        print("="*80)
        print("Please check if log.txt files exist in the output directory.")

if __name__ == "__main__":
    main()