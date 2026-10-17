import numpy as np
import matplotlib.pyplot as plt

np.random.seed(555)
fig2 = plt.figure(figsize=(10, 13))

# Top Section: Unique English Attention Weight Distribution Mapping
ax_top = plt.subplot2grid((4, 2), (0, 0), colspan=2)
english_tokens = ['The', 'patient', 'is', '[pause]', 'trying', 'to', 'explain', 'the', '[long-pause]', 'picture']
attention_scores = [0.04, 0.05, 0.04, 0.35, 0.06, 0.04, 0.05, 0.03, 0.31, 0.03] 
bar_colors = ['#00A8B5' if score < 0.1 else '#0F2240' for score in attention_scores]

ax_top.barh(english_tokens[::-1], attention_scores[::-1], color=bar_colors[::-1], height=0.5)
ax_top.set_title('Gated Transformer Attention Magnitudes (Clinical English Context)', fontsize=11, fontweight='bold', color='#0F2240')
ax_top.set_xlabel('Softmax Weight Assignment')
ax_top.set_xlim(0, 0.5)
ax_top.grid(axis='x', alpha=0.2)

# Row 2, Left: Diagnostic Boundary Extrapolations
ax_mid_left = plt.subplot2grid((4, 2), (1, 0), rowspan=2)
range_val = np.linspace(0, 6, 120)
ctrl_curve = np.exp(-range_val/0.5) * 6
deficit_curve = (range_val**2.5) * np.exp(-range_val/0.55)

ax_mid_left.plot(range_val, ctrl_curve, color='#00A8B5', label='Asymptomatic Cohort', lw=2)
ax_mid_left.plot(range_val, deficit_curve, color='#0F2240', label='Cognitive Shift Cohort', lw=2)
ax_mid_left.set_title('A) Diagnostic Boundary Extraction', fontsize=9, fontweight='bold', color='#0F2240')
ax_mid_left.set_xlabel('Calculated Index Boundary')
ax_mid_left.set_ylabel('Normalized Concentration')
ax_mid_left.legend()
ax_mid_left.grid(True, alpha=0.1)

# Row 2, Right: Speech Velocity Metrics over Time
ax_mid_right = plt.subplot2grid((4, 2), (1, 1), rowspan=2)
timeline = np.linspace(0, 15, 150)
v_ctrl = 4.2 + 0.25 * np.cos(timeline/1.5) + np.random.normal(0, 0.04, 150)
v_deficit = 2.2 - 0.08 * timeline + 0.5 * np.sin(timeline * 1.2) * np.random.uniform(0.4, 1.4, 150)

ax_mid_right.plot(timeline, v_ctrl, color='#00A8B5', alpha=0.85, label='Control')
ax_mid_right.plot(timeline, v_deficit, color='#0F2240', alpha=0.85, label='Impaired')
ax_mid_right.set_title('B) Articulatory Speech Velocity Delta', fontsize=9, fontweight='bold', color='#0F2240')
ax_mid_right.set_xlabel('Window Index Time (s)')
ax_mid_right.set_ylabel('Velocity Derivative')
ax_mid_right.grid(True, alpha=0.1)

# Row 3: Global Feature System Importance (SHAP Framework)
ax_bottom = plt.subplot2grid((4, 2), (3, 0), colspan=2)
upgraded_features = ['Cross-Modal Synthesis Nodes', 'Conformer Structural Latents', 'ClinicalBERT Token Metrics', 'Denoised Signal Variance', 'Fundamental Pitch Modulations']
impact_metrics = [0.34, 0.24, 0.19, 0.13, 0.10]

ax_bottom.barh(upgraded_features[::-1], impact_metrics[::-1], color='#00A8B5', alpha=0.9, height=0.45)
ax_bottom.set_title('C) Global Pipeline Component Weights (SHAP Evaluation Values)', fontsize=9, fontweight='bold', color='#0F2240')
ax_bottom.set_xlabel('Mean Absolute Weight Metric')
ax_bottom.grid(axis='x', alpha=0.1)

plt.tight_layout()
plt.savefig('unique_clinical_metrics_english.png', dpi=300)
plt.show()