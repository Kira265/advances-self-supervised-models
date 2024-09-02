import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import seaborn as sns
from thop import profile
import pandas as pd

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = torch.matmul(q, k.transpose(-2, -1) / (q.size(-1) ** 0.5))
        attn = torch.softmax(attn, dim=-1)

        return torch.matmul(attn, v)
    
def profile_self_attention(seq_len, dim, device, num_runs=10):
    model = SelfAttention(dim).to(device)
    x = torch.randn(1, seq_len, dim).to(device)

    # Measure FLOPS
    flops, _ = profile(model, inputs=(x,))

    # Measure memory usage
    torch.cuda.reset_peak_memory_stats()
    model(x)
    memory_usage = torch.cuda.max_memory_allocated() / 1024**2

    # Measure wall clock time
    start_time = time.time()
    for _ in range(num_runs):
        model(x)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs

    return flops, memory_usage, avg_time

def run_experiments(seq_lengths, dim, device, num_runs=10, num_trials=5):
    results = []

    for seq_len in seq_lengths:
        for _ in range(num_trials):
            flops, memory, avg_time = profile_self_attention(seq_len, dim, device, num_runs)
            results.append({
                'seq_len': seq_len,
                'flops': flops,
                'memory': memory,
                'time': avg_time
            })
    
    return pd.DataFrame(results)

def plot_results(results_gpu, results_cpu):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    metrics = ['flops', 'memory', 'time']
    titles = ['FLOPS', 'Memory Usage (MB)', 'Wall Clock Time (s)']

    for i, (metric, title) in enumerate(metrics, titles):
        sns.lineplot(x='seq_len', y=metric, data=results_gpu, ax=axs[i], label='GPU', marker='o', errorbar='se')
        sns.lineplot(x='seq_len', y=metric, data=results_cpu, ax=axs[i], label='CPU', marker='o', errorbar='se')

        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        set[i].set_xlabel('Sequence Length')
        axs[i].set_ylabel(title)
        axs[i].set_title(f'{title} vs Sequence Length')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig('self_attention_profile.png')
    plt.close()

if __name__ == "__main__":
    seq_lengths = [10, 100, 1000, 10000, 100000]
    dim = 64

    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    results_gpu = run_experiments(seq_lengths, dim, device_gpu)
    results_cpu = run_experiments(seq_lengths, dim, device_cpu)

    plot_results(results_gpu, results_cpu)

    print("Profiling complete. Results saved in 'self_attention_profile.png'.")

