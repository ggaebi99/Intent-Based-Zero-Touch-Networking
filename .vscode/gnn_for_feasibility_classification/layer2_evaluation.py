import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# ---------------------------------------------------------
# 1. 임의의 테스트 데이터 및 예측 결과 생성 (실제 데이터로 교체하세요!)
# ---------------------------------------------------------
np.random.seed(42)
y_true = np.random.randint(0, 2, 200) # 실제 정답 (0: Infeasible, 1: Feasible)

# MLP 예측 확률 (성능이 살짝 떨어지게 세팅)
mlp_probs = np.clip(y_true * 0.6 + np.random.normal(0, 0.4, 200), 0, 1)
mlp_preds = (mlp_probs > 0.5).astype(int)

# GINE 예측 확률 (성능이 더 우수하게 세팅)
gine_probs = np.clip(y_true * 0.85 + np.random.normal(0, 0.2, 200), 0, 1)
gine_preds = (gine_probs > 0.5).astype(int)

# ---------------------------------------------------------
# 2. 성능 비교 표 (Pandas DataFrame) 출력
# ---------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        'Accuracy': report['accuracy'],
        'Precision': report['macro avg']['precision'],
        'Recall': report['macro avg']['recall'],
        'F1-Score': report['macro avg']['f1-score']
    }

metrics_mlp = calculate_metrics(y_true, mlp_preds)
metrics_gine = calculate_metrics(y_true, gine_preds)

df_results = pd.DataFrame([metrics_mlp, metrics_gine], index=['Baseline (MLP)', 'Proposed (GINE)'])
df_results = df_results.round(4) * 100 # 보기 좋게 퍼센트로 변환

print("=== Layer 2: Feasibility Prediction Performance (%) ===")
print(df_results)
print("=" * 55)

# ---------------------------------------------------------
# 3. 학술적 톤의 오차 행렬 & ROC Curve 시각화
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (1) Proposed Model (GINE) 오차 행렬 (Confusion Matrix)
cm = confusion_matrix(y_true, gine_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            cbar=False, ax=axes[0], annot_kws={"size": 16, "weight": "bold"})

axes[0].set_title('(a) Confusion Matrix of Proposed GINE', fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=14, fontweight='bold')
axes[0].set_xticklabels(['Infeasible (0)', 'Feasible (1)'], fontsize=12)
axes[0].set_yticklabels(['Infeasible (0)', 'Feasible (1)'], fontsize=12, va='center')

# (2) ROC Curve (MLP vs GINE 비교)
fpr_mlp, tpr_mlp, _ = roc_curve(y_true, mlp_probs)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

fpr_gine, tpr_gine, _ = roc_curve(y_true, gine_probs)
roc_auc_gine = auc(fpr_gine, tpr_gine)

axes[1].plot(fpr_mlp, tpr_mlp, color='#F39C12', lw=2.5, linestyle='--', label=f'Baseline MLP (AUC = {roc_auc_mlp:.3f})')
axes[1].plot(fpr_gine, tpr_gine, color='#4A90E2', lw=3, label=f'Proposed GINE (AUC = {roc_auc_gine:.3f})')
axes[1].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle=':') # 50% 확률의 기준선

axes[1].set_title('(b) ROC Curve Comparison', fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
axes[1].tick_params(labelsize=12)
axes[1].legend(loc="lower right", fontsize=12, frameon=True, shadow=True)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(top=0.90, wspace=0.25)

# 이미지 저장
output_filename = 'layer_2_feasibility_evaluation.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"✅ 오차 행렬과 ROC Curve가 통합된 고해상도 이미지가 '{output_filename}' 이름으로 저장되었습니다!")
plt.show()