import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import os

def run_benchmarks():
    print("Running Prefix-Prefill Benchmarks...")
    # Assuming this script is run from the triton_ops root directory
    result = subprocess.run(['python', 'benchmarks/bench_prefix_prefill.py'], capture_output=True, text=True)
    print(result.stdout)
    
    lines = result.stdout.strip().split('\n')
    data = []
    
    for line in lines:
        if 'ms' in line and '%' in line:
            parts = [p.strip() for p in line.split('|')]
            P = int(parts[0])
            S = int(parts[1])
            std_ms = float(parts[2].replace('ms', ''))
            prefix_ms = float(parts[3].replace('ms', ''))
            red = float(parts[4].replace('%', ''))
            data.append({
                'Scenario': f"P={P}, S={S}",
                'Standard (ms)': std_ms,
                'Prefix-Prefill (ms)': prefix_ms,
                'Reduction (%)': red
            })
            
    if not data:
        print("No benchmark data parsed. Ensure bench_prefix_prefill.py outputs correctly.")
        return

    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], df['Standard (ms)'], width, label='Standard TTFT', color='#ff9999')
    ax.bar([i + width/2 for i in x], df['Prefix-Prefill (ms)'], width, label='Prefix-Prefill TTFT', color='#66b3ff')
    
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('TTFT Reduction on Google Colab (T4 GPU)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Scenario'], fontsize=10)
    ax.legend()
    
    # Add reduction % text
    for i, row in df.iterrows():
        ax.text(i + width/2, row['Prefix-Prefill (ms)'] + max(df['Standard (ms)'])*0.02, 
                f"-{row['Reduction (%)']}%", ha='center', color='navy', fontweight='bold')
    
    plt.tight_layout()
    plot_path = 'benchmarks/colab_ttft_results.png'
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot successfully saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_benchmarks()
