"""
Executes all notebook logic to produce charts + processed data files.
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score,
                              precision_score, recall_score)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120

print("="*55)
print("  CUSTOMER CHURN PREDICTION — FULL PIPELINE RUN")
print("="*55)

# ── STEP 1: Load raw data ────────────────────────────────────────────
df = pd.read_csv('data/telco_churn.csv')
print(f"\n[1/6] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"      Churn rate: {(df['Churn']=='Yes').mean()*100:.2f}%")

# ── STEP 2: EDA Charts ────────────────────────────────────────────────
print("\n[2/6] Generating EDA charts...")

# Chart 1: Churn distribution
churn_counts = df['Churn'].value_counts()
churn_pct    = df['Churn'].value_counts(normalize=True) * 100
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(churn_counts.index, churn_counts.values,
            color=['#2ecc71','#e74c3c'], edgecolor='white', linewidth=1.5)
axes[0].set_title('Churn Count Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Churn'); axes[0].set_ylabel('Customers')
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v+50, str(v), ha='center', fontweight='bold')
axes[1].pie(churn_pct.values, labels=churn_pct.index, autopct='%1.1f%%',
            colors=['#2ecc71','#e74c3c'], startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1].set_title('Churn Rate Breakdown', fontsize=14, fontweight='bold')
plt.suptitle('Customer Churn Overview', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('charts/01_churn_distribution.png', bbox_inches='tight')
plt.close()

# Chart 2: Numerical distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ['tenure','MonthlyCharges','TotalCharges']):
    ax.hist(df[df['Churn']=='No'][col],  bins=30, alpha=0.6, color='#2ecc71', label='Not Churned', edgecolor='white')
    ax.hist(df[df['Churn']=='Yes'][col], bins=30, alpha=0.6, color='#e74c3c', label='Churned',     edgecolor='white')
    ax.set_title(f'{col} by Churn', fontweight='bold')
    ax.set_xlabel(col); ax.legend()
plt.suptitle('Numerical Features by Churn Status', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/02_numerical_distributions.png', bbox_inches='tight')
plt.close()

# Chart 3: Churn by contract
contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean()*100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(contract_churn.index, contract_churn.values,
              color=['#e74c3c','#f39c12','#2ecc71'], edgecolor='white', linewidth=1.5)
ax.set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
for bar, val in zip(bars, contract_churn.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/03_churn_by_contract.png', bbox_inches='tight')
plt.close()

# Chart 4: Churn by internet & payment
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, title in zip(axes,
    ['InternetService','PaymentMethod'],
    ['Churn Rate by Internet Service','Churn Rate by Payment Method']):
    rates = df.groupby(col)['Churn'].apply(lambda x: (x=='Yes').mean()*100).sort_values()
    bars = ax.barh(rates.index, rates.values, color='#3498db', edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Churn Rate (%)')
    for bar, val in zip(bars, rates.values):
        ax.text(val+0.3, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/04_churn_by_service_payment.png', bbox_inches='tight')
plt.close()

# Chart 5: Scatter
fig, ax = plt.subplots(figsize=(10, 6))
for label, color in [('No','#2ecc71'),('Yes','#e74c3c')]:
    g = df[df['Churn']==label]
    ax.scatter(g['tenure'], g['MonthlyCharges'], c=color, alpha=0.35, s=15, label=f'Churn: {label}')
ax.set_title('Tenure vs Monthly Charges by Churn', fontsize=14, fontweight='bold')
ax.set_xlabel('Tenure (months)'); ax.set_ylabel('Monthly Charges ($)')
ax.legend()
plt.tight_layout()
plt.savefig('charts/05_tenure_vs_charges_scatter.png', bbox_inches='tight')
plt.close()
print("      Charts 01–05 saved.")

# ── STEP 3: Cleaning & Feature Engineering ───────────────────────────
print("\n[3/6] Cleaning & feature engineering...")

df2 = df.copy()
df2.drop(columns=['customerID'], inplace=True)
df2['TotalCharges'] = pd.to_numeric(df2['TotalCharges'], errors='coerce')
df2['TotalCharges'].fillna(df2['TotalCharges'].median(), inplace=True)
df2['Churn'] = (df2['Churn'] == 'Yes').astype(int)

df2['TenureGroup']     = pd.cut(df2['tenure'], bins=[0,12,24,48,72], labels=['0-1yr','1-2yr','2-4yr','4+yr'])
df2['ChargePerTenure'] = (df2['MonthlyCharges'] / (df2['tenure'] + 1)).round(2)
service_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
df2['NumServices'] = df2[service_cols].apply(lambda r: sum(v=='Yes' for v in r), axis=1)
df2['HighValue']   = (df2['MonthlyCharges'] > df2['MonthlyCharges'].quantile(0.75)).astype(int)

for col in ['gender','Partner','Dependents','PhoneService','PaperlessBilling']:
    df2[col] = LabelEncoder().fit_transform(df2[col])

multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
              'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
              'Contract','PaymentMethod','TenureGroup']
df2 = pd.get_dummies(df2, columns=multi_cols, drop_first=True)

# Chart 6: Correlation heatmap
top_feats = ['tenure','MonthlyCharges','TotalCharges','NumServices','ChargePerTenure','HighValue','Churn']
corr = df2[top_feats].corr()
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            mask=mask, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/06_correlation_heatmap.png', bbox_inches='tight')
plt.close()

# Oversample minority class
majority  = df2[df2['Churn']==0]
minority  = df2[df2['Churn']==1]
minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
df_bal    = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)

df2.to_csv('data/telco_churn_cleaned.csv', index=False)
df_bal.to_csv('data/telco_churn_balanced.csv', index=False)
print(f"      Cleaned: {df2.shape} | Balanced: {df_bal.shape}")

# ── STEP 4: Model Building ───────────────────────────────────────────
print("\n[4/6] Training models...")

X = df_bal.drop(columns=['Churn']); y = df_bal['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
}
results = {}
for name, model in models.items():
    Xtr = X_train_sc if name=='Logistic Regression' else X_train
    Xte = X_test_sc  if name=='Logistic Regression' else X_test
    model.fit(Xtr, y_train)
    yp  = model.predict(Xte)
    ypr = model.predict_proba(Xte)[:,1]
    results[name] = dict(model=model, y_pred=yp, y_prob=ypr,
        precision=precision_score(y_test,yp), recall=recall_score(y_test,yp),
        f1=f1_score(y_test,yp), roc_auc=roc_auc_score(y_test,ypr), Xte=Xte)
    print(f"      {name}: F1={results[name]['f1']:.3f}  AUC={results[name]['roc_auc']:.3f}")

# Chart 7: Model comparison
metrics = ['precision','recall','f1','roc_auc']
colors  = ['#3498db','#2ecc71','#e74c3c']
x = np.arange(len(metrics)); width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    bars = ax.bar(x + i*width, vals, width, label=name, color=colors[i], edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.2f}', ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x+width); ax.set_xticklabels(['Precision','Recall','F1','ROC-AUC'], fontsize=11)
ax.set_ylim(0, 1.12); ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold'); ax.legend()
plt.tight_layout()
plt.savefig('charts/07_model_comparison.png', bbox_inches='tight')
plt.close()

# Chart 8: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Churned','Churned'], yticklabels=['Not Churned','Churned'],
                linewidths=1, linecolor='white')
    ax.set_title(f'{name}\nF1:{res["f1"]:.3f} | AUC:{res["roc_auc"]:.3f}', fontweight='bold', fontsize=10)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('charts/08_confusion_matrices.png', bbox_inches='tight')
plt.close()

# Chart 9: ROC curves
fig, ax = plt.subplots(figsize=(9, 7))
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC={res["roc_auc"]:.3f})')
ax.plot([0,1],[0,1],'k--', lw=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12); ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/09_roc_curves.png', bbox_inches='tight')
plt.close()

# Chart 10: Feature importance
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=X_train.columns).nlargest(15).sort_values()
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(importances.index, importances.values, color='#3498db', edgecolor='white')
for bar, val in zip(bars, importances.values):
    ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
ax.set_title('Top 15 Feature Importances — Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('charts/10_feature_importance.png', bbox_inches='tight')
plt.close()
print("      Charts 06–10 saved.")

# ── STEP 5: Risk Scoring ──────────────────────────────────────────────
print("\n[5/6] Generating risk scores...")

gb = results['Gradient Boosting']['model']
X_full = df2.drop(columns=['Churn']).reindex(columns=X_train.columns, fill_value=0)
probs = gb.predict_proba(X_full)[:,1]
df2['ChurnProbability'] = probs
df2['RiskScore']        = (probs * 100).round(1)

def tier(s):
    if s < 25:   return 'Low Risk'
    elif s < 50: return 'Medium Risk'
    elif s < 75: return 'High Risk'
    else:        return 'Critical Risk'

df2['RiskTier'] = df2['RiskScore'].apply(tier)
tier_order  = ['Low Risk','Medium Risk','High Risk','Critical Risk']
tier_colors = ['#2ecc71','#f39c12','#e74c3c','#8e44ad']

# Attach risk back to original df
orig = df.copy()
orig['RiskScore'] = df2['RiskScore'].values
orig['RiskTier']  = df2['RiskTier'].values
orig.to_csv('data/telco_churn_risk_scored.csv', index=False)

# Chart 11: Risk tier
counts = [len(df2[df2['RiskTier']==t]) for t in tier_order]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
bars = axes[0].bar(tier_order, counts, color=tier_colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Customers by Risk Tier', fontsize=14, fontweight='bold'); axes[0].set_ylabel('Customers')
for bar, val in zip(bars, counts):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+30, str(val), ha='center', fontweight='bold')
axes[1].pie(counts, labels=tier_order, autopct='%1.1f%%', colors=tier_colors, startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1].set_title('Risk Tier Distribution', fontsize=14, fontweight='bold')
plt.suptitle('Churn Risk Tier Segmentation', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/11_risk_tier_distribution.png', bbox_inches='tight')
plt.close()

# Chart 12: Risk score histogram
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(df2['RiskScore'], bins=50, color='#3498db', edgecolor='white', alpha=0.85)
for thresh, col, lbl in [(25,'#2ecc71','Low'),(50,'#f39c12','Med'),(75,'#e74c3c','High')]:
    ax.axvline(thresh, color=col, lw=2, linestyle='--', label=f'{lbl} threshold ({thresh})')
ax.set_title('Churn Risk Score Distribution (0–100)', fontsize=14, fontweight='bold')
ax.set_xlabel('Risk Score'); ax.set_ylabel('Count'); ax.legend()
plt.tight_layout()
plt.savefig('charts/12_risk_score_distribution.png', bbox_inches='tight')
plt.close()

# Chart 13: Metrics by risk tier
avg = orig.groupby('RiskTier')[['tenure','MonthlyCharges','TotalCharges']].mean().loc[tier_order]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ['tenure','MonthlyCharges','TotalCharges']):
    bars = ax.bar(avg.index, avg[col], color=tier_colors, edgecolor='white')
    ax.set_title(f'Avg {col}', fontweight='bold')
    ax.set_xticklabels(avg.index, rotation=20, ha='right', fontsize=9)
    for bar, val in zip(bars, avg[col]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')
plt.suptitle('Customer Metrics by Risk Tier', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/13_metrics_by_risk_tier.png', bbox_inches='tight')
plt.close()
print("      Charts 11–13 saved.")

# ── STEP 6: Executive Dashboard ───────────────────────────────────────
print("\n[6/6] Building executive dashboard...")

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#1a1a2e')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
t_s = dict(color='white', fontsize=11, fontweight='bold', pad=8)

hi_risk_rev = orig[orig['RiskTier'].isin(['High Risk','Critical Risk'])]['MonthlyCharges'].sum()
kpis = [
    ("Total Customers",    f"{len(orig):,}",               "#3498db"),
    ("Churn Rate",         f"{(orig['Churn']=='Yes').mean()*100:.1f}%", "#e74c3c"),
    ("Monthly Revenue\nat Risk", f"${hi_risk_rev:,.0f}",   "#8e44ad"),
]
for idx, (lbl, val, col) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, idx])
    ax.set_facecolor(col)
    ax.text(0.5, 0.62, val, ha='center', va='center',
            fontsize=22, fontweight='bold', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.22, lbl, ha='center', va='center',
            fontsize=9, color='white', transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])

ax1 = fig.add_subplot(gs[1, 0]); ax1.set_facecolor('#16213e')
cc = orig.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean()*100)
bars = ax1.bar(cc.index, cc.values, color=['#e74c3c','#f39c12','#2ecc71'], edgecolor='none')
ax1.set_title('Churn Rate by Contract', **t_s)
ax1.tick_params(colors='white', labelsize=8); ax1.spines[:].set_visible(False)
for bar, v in zip(bars, cc.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{v:.0f}%',
             ha='center', color='white', fontsize=8, fontweight='bold')

ax2 = fig.add_subplot(gs[1, 1]); ax2.set_facecolor('#16213e')
tc = [len(orig[orig['RiskTier']==t]) for t in tier_order]
wedges, texts, autotexts = ax2.pie(tc, labels=tier_order, autopct='%1.1f%%',
    colors=tier_colors, startangle=90, wedgeprops=dict(edgecolor='#1a1a2e', linewidth=2))
for t in texts:      t.set_color('white'); t.set_fontsize(7)
for at in autotexts: at.set_color('white'); at.set_fontsize(7)
ax2.set_title('Risk Tier Split', **t_s)

ax3 = fig.add_subplot(gs[1, 2]); ax3.set_facecolor('#16213e')
for lbl, col in [('No','#2ecc71'),('Yes','#e74c3c')]:
    ax3.hist(orig[orig['Churn']==lbl]['MonthlyCharges'],
             bins=25, alpha=0.7, color=col, label=f'Churn: {lbl}', edgecolor='none')
ax3.set_title('Monthly Charges by Churn', **t_s)
ax3.tick_params(colors='white', labelsize=8)
ax3.legend(fontsize=7, facecolor='#16213e', labelcolor='white')
ax3.spines[:].set_visible(False)

ax4 = fig.add_subplot(gs[2, :2]); ax4.set_facecolor('#16213e')
cmap_t = {'Low Risk':'#2ecc71','Medium Risk':'#f39c12','High Risk':'#e74c3c','Critical Risk':'#8e44ad'}
for t in tier_order:
    sub = orig[orig['RiskTier']==t]
    ax4.scatter(sub['tenure'], sub['RiskScore'], c=cmap_t[t], alpha=0.3, s=10, label=t)
ax4.set_title('Tenure vs Churn Risk Score', **t_s)
ax4.set_xlabel('Tenure (months)', color='white', fontsize=9)
ax4.set_ylabel('Risk Score (0–100)', color='white', fontsize=9)
ax4.tick_params(colors='white', labelsize=8)
ax4.legend(fontsize=7, facecolor='#16213e', labelcolor='white', markerscale=2)
ax4.spines[:].set_visible(False)

ax5 = fig.add_subplot(gs[2, 2]); ax5.set_facecolor('#16213e')
ic = orig.groupby('InternetService')['Churn'].apply(lambda x: (x=='Yes').mean()*100).sort_values()
ax5.barh(ic.index, ic.values, color=['#2ecc71','#f39c12','#e74c3c'], edgecolor='none')
ax5.set_title('Churn by Internet Service', **t_s)
ax5.tick_params(colors='white', labelsize=8); ax5.spines[:].set_visible(False)
for i, v in enumerate(ic.values):
    ax5.text(v+0.3, i, f'{v:.1f}%', va='center', color='white', fontsize=8, fontweight='bold')

fig.suptitle('Customer Churn Prediction — Executive Dashboard',
             color='white', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('charts/14_final_summary_dashboard.png', bbox_inches='tight', facecolor='#1a1a2e')
plt.close()

# Statistical summary CSV
summary = pd.DataFrame({
    'Metric': [
        'Total Customers','Churned Customers','Churn Rate (%)',
        'Avg Tenure Churned (mo)','Avg Tenure Retained (mo)',
        'Avg Monthly Charge Churned','Avg Monthly Charge Retained',
        'Low Risk Customers','Medium Risk Customers',
        'High Risk Customers','Critical Risk Customers',
        'Monthly Revenue at Risk ($)',
    ],
    'Value': [
        len(orig), (orig['Churn']=='Yes').sum(),
        round((orig['Churn']=='Yes').mean()*100, 2),
        round(orig[orig['Churn']=='Yes']['tenure'].mean(), 1),
        round(orig[orig['Churn']=='No']['tenure'].mean(), 1),
        round(orig[orig['Churn']=='Yes']['MonthlyCharges'].mean(), 2),
        round(orig[orig['Churn']=='No']['MonthlyCharges'].mean(), 2),
        len(orig[orig['RiskTier']=='Low Risk']),
        len(orig[orig['RiskTier']=='Medium Risk']),
        len(orig[orig['RiskTier']=='High Risk']),
        len(orig[orig['RiskTier']=='Critical Risk']),
        round(hi_risk_rev, 2),
    ]
})
summary.to_csv('charts/statistical_summary.csv', index=False)

print("      Charts 14 + statistical_summary.csv saved.")
print("\n✅ All charts generated successfully!")
print(f"\n{'─'*45}")
print("  FINAL MODEL RESULTS")
print(f"{'─'*45}")
for name, res in results.items():
    print(f"  {name:<25} F1:{res['f1']:.3f}  AUC:{res['roc_auc']:.3f}")
print(f"{'─'*45}")
hi = orig[orig['RiskTier'].isin(['High Risk','Critical Risk'])]
print(f"  High+Critical Risk Customers : {len(hi):,}")
print(f"  Monthly Revenue at Risk      : ${hi['MonthlyCharges'].sum():,.2f}")
print(f"{'─'*45}")

import joblib

joblib.dump(results['Gradient Boosting']['model'], 'api/model.pkl')
joblib.dump(X_train.columns.tolist(), 'api/features.pkl')