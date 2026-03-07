import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_results(filename):
    processes = []
    exec_times = []
    speedups = []
    efficiencies = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if '|' in line and not 'Processes' in line and not '---' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                try:
                    processes.append(int(parts[0]))
                    exec_times.append(float(parts[1]))
                    speedups.append(float(parts[2]))
                    efficiencies.append(float(parts[3]))
                except ValueError:
                    continue
    
    return processes, exec_times, speedups, efficiencies

def plot_performance_metrics(processes, exec_times, speedups, efficiencies):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MPI Document Summarizer Performance Analysis', fontsize=16, fontweight='bold')
    
    # Execution Time
    axes[0, 0].plot(processes, exec_times, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('Number of Processes', fontsize=11)
    axes[0, 0].set_ylabel('Execution Time (seconds)', fontsize=11)
    axes[0, 0].set_title('Execution Time vs Number of Processes', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(processes)
    
    # Speedup
    axes[0, 1].plot(processes, speedups, marker='s', linewidth=2, markersize=8, color='#A23B72', label='Actual Speedup')
    ideal_speedup = [1] + [p-1 for p in processes[1:]]
    axes[0, 1].plot(processes, ideal_speedup, linestyle='--', linewidth=2, color='#F18F01', label='Ideal Speedup')
    axes[0, 1].set_xlabel('Number of Processes', fontsize=11)
    axes[0, 1].set_ylabel('Speedup', fontsize=11)
    axes[0, 1].set_title('Speedup vs Number of Processes', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(processes)
    
    # Efficiency
    axes[1, 0].plot(processes, efficiencies, marker='^', linewidth=2, markersize=8, color='#C73E1D')
    axes[1, 0].axhline(y=1.0, linestyle='--', color='gray', linewidth=1, label='100% Efficiency')
    axes[1, 0].set_xlabel('Number of Processes', fontsize=11)
    axes[1, 0].set_ylabel('Efficiency', fontsize=11)
    axes[1, 0].set_title('Parallel Efficiency vs Number of Processes', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(processes)
    axes[1, 0].set_ylim([0, max(efficiencies) * 1.2])
    
    # Bar chart comparison
    x = np.arange(len(processes))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, speedups, width, label='Speedup', color='#2E86AB', alpha=0.8)
    axes[1, 1].bar(x + width/2, efficiencies, width, label='Efficiency', color='#A23B72', alpha=0.8)
    axes[1, 1].set_xlabel('Number of Processes', fontsize=11)
    axes[1, 1].set_ylabel('Value', fontsize=11)
    axes[1, 1].set_title('Speedup and Efficiency Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(processes)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('mpi_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Performance analysis plot saved as 'mpi_performance_analysis.png'")
    plt.show()

def print_analysis(processes, exec_times, speedups, efficiencies):
    print("\n" + "="*60)
    print("MPI PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nBest Speedup: {max(speedups):.2f}x at {processes[speedups.index(max(speedups))]} processes")
    print(f"Best Efficiency: {max(efficiencies):.2f} at {processes[efficiencies.index(max(efficiencies))]} processes")
    
    print(f"\nFastest Execution: {min(exec_times):.4f}s at {processes[exec_times.index(min(exec_times))]} processes")
    print(f"Slowest Execution: {max(exec_times):.4f}s at {processes[exec_times.index(max(exec_times))]} processes")
    
    avg_efficiency = sum(efficiencies) / len(efficiencies)
    print(f"\nAverage Efficiency: {avg_efficiency:.2f}")
    
    print("\nScalability Analysis:")
    for i in range(1, len(processes)):
        time_reduction = ((exec_times[i-1] - exec_times[i]) / exec_times[i-1]) * 100
        print(f"  {processes[i-1]} → {processes[i]} processes: {time_reduction:.1f}% time reduction")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mpi_metrics.py <performance_results.txt>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    try:
        processes, exec_times, speedups, efficiencies = parse_results(results_file)
        
        if not processes:
            print("Error: No valid data found in results file")
            sys.exit(1)
        
        print_analysis(processes, exec_times, speedups, efficiencies)
        plot_performance_metrics(processes, exec_times, speedups, efficiencies)
        
    except FileNotFoundError:
        print(f"Error: File '{results_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
