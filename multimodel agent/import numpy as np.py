import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Unique seed to ensure your data distribution looks completely different
np.random.seed(444)

fig1, axs1 = plt.subplots(2, 2, figsize=(11, 8.5))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Plot A: Cleaned WavLM Spectrum Envelope
time_axis = np.linspace(0, 3.0, 1200)
wavlm_signal = 0.85 * np.cos(2 * np.pi * 40 * time_axis) * np.exp(-time_axis * 0.4) + np.random.normal(0, 0.02, 1200)
axs1[0, 0].plot(time_axis, wavlm_signal, color='#0F2240', lw=1.2)
axs1[0, 0].set_title('A) WavLM Denoised Signal Waveform', fontsize=10, fontweight='bold', color='#0F2240')
axs1[0, 0].set_xlabel('Time Frame (s)', fontsize=8)
axs1[0, 0].set_ylabel('Amplitude (Normalized)', fontsize=8)
axs1[0, 0].grid(True, alpha=0.15)

# Plot B: Conformer Latent Feature Map (256 Channels)
conformer_matrix = np.random.uniform(0.0, 1.0, (256, 110))
sns.heatmap(conformer_matrix, ax=axs1[0, 1], cmap='viridis', cbar=False) 
axs1[0, 1].set_title('B) Conformer Encoder Extracted Features', fontsize=10, fontweight='bold', color='#0F2240')
axs1[0, 1].set_xlabel('Acoustic Windows', fontsize=8)
axs1[0, 1].set_ylabel('Latent Matrix Channels (256)', fontsize=8)

# Plot C: ClinicalBERT Token Space Matrix (768 Hidden Dimensions)
text_matrix = np.random.normal(0.0, 1.2, (768, 90))
sns.heatmap(text_matrix, ax=axs1[1, 0], cmap='magma', cbar=False) 
axs1[1, 0].set_title('C) ClinicalBERT Text Token Dimensions', fontsize=10, fontweight='bold', color='#0F2240')
axs1[1, 0].set_xlabel('Linguistic Time-steps', fontsize=8)
axs1[1, 0].set_ylabel('Embedding Space Tensor (768)', fontsize=8)

# Plot D: Cross-Modal Scaled Dot-Product Alignment Map
cross_attention_weights = np.random.uniform(0.1, 0.4, (80, 80))
for idx in range(80):
    cross_attention_weights[idx, idx] += 0.5
    if idx % 10 == 0:
        cross_attention_weights[max(0, idx-4):min(80, idx+6), idx] += 0.35
sns.heatmap(cross_attention_weights, ax=axs1[1, 1], cmap='icefire', vmin=0.1, vmax=1.1) 
axs1[1, 1].set_title('D) Cross-Modal Token-Audio Attention Map', fontsize=10, fontweight='bold', color='#0F2240')
axs1[1, 1].set_xlabel('Linguistic Key (ClinicalBERT)', fontsize=8)
axs1[1, 1].set_ylabel('Acoustic Query (Conformer)', fontsize=8)

plt.tight_layout()
plt.savefig('unique_model_streams_english.png', dpi=300)
plt.show()