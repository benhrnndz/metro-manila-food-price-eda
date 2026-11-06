import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Setup styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'

os.makedirs('images', exist_ok=True)

# 1. Load Data
df = pd.read_csv('weekly_prices_ncr.csv')
df['week_start'] = pd.to_datetime(df['week_start'])
df['week_end'] = pd.to_datetime(df['week_end'])
df['price_num'] = pd.to_numeric(df['price'], errors='coerce')

# Fix known typos in commodity_std
typo_map = {
    'Basmaf': 'Rice, Imported - Basmati',
    'Basmati': 'Rice, Imported - Basmati',
    'Japorica Jasoonica': 'Rice, Imported - Japonica Jasponica',
    'Glutinous': 'Rice, Local - Glutinous'
}
df['commodity_std'] = df['commodity_std'].replace(typo_map)

# FIG 1: Dataset Overview & Quality
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1A: Records per category
cat_counts = df['category'].value_counts()
sns.barplot(x=cat_counts.values, y=cat_counts.index, ax=axs[0], palette='Blues_r')
axs[0].set_title('A. Records by Category (Total: 13,387)', fontsize=12, fontweight='bold')
axs[0].set_xlabel('Record Count')
for i, v in enumerate(cat_counts.values):
    axs[0].text(v + 30, i, f'{v:,}', va='center', fontsize=9)

# 1B: Data Extraction Method
method_counts = df['method'].value_counts()
colors = ['#4575b4', '#91bfdb']
axs[1].pie(method_counts, labels=method_counts.index, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0.05, 0))
axs[1].set_title('B. Extraction Method Breakdown', fontsize=12, fontweight='bold')

# 1C: Missing Prices by Category
missing_by_cat = df[df['price_num'].isna()]['category'].value_counts()
sns.barplot(x=missing_by_cat.values, y=missing_by_cat.index, ax=axs[2], palette='Reds_r')
axs[2].set_title('C. Missing / Unparseable Prices by Category (Total: 576)', fontsize=12, fontweight='bold')
axs[2].set_xlabel('Missing Price Count')
for i, v in enumerate(missing_by_cat.values):
    axs[2].text(v + 5, i, f'{v}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('images/fig1_dataset_overview.png', dpi=300)
plt.close()
print('Fig 1 saved.')

# 2. Reindexed continuous weekly series for Well-Milled Rice
wm_df = df[df['commodity_std'] == 'Rice, Local - Well Milled'].sort_values('week_start').set_index('week_start')
full_idx = pd.date_range(start=wm_df.index.min(), end=wm_df.index.max(), freq='W-MON')
wm_ts = wm_df['price_num'].reindex(full_idx).interpolate(method='linear')

# Calculate indicators
wm_roll_mean = wm_ts.rolling(12, min_periods=4).mean()
wm_roll_std = wm_ts.rolling(12, min_periods=4).std()
wm_bb_upper = wm_roll_mean + 2 * wm_roll_std
wm_bb_lower = wm_roll_mean - 2 * wm_roll_std

# 4-week percentage change
wm_pct4 = wm_ts.pct_change(4) * 100
wm_spikes = wm_pct4 > 5.0

# FIG 2: Primary Anchor: Well Milled Rice & Spike Detection
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})

# Top panel: Price and Bollinger Bands
ax1.plot(wm_ts.index, wm_ts, label='Retail Price (₱/kg)', color='#1b7837', lw=2)
ax1.plot(wm_roll_mean.index, wm_roll_mean, label='12-Week Rolling Mean', color='#762a83', lw=1.5, ls='--')
ax1.fill_between(wm_ts.index, wm_bb_lower, wm_bb_upper, color='#e7d4e8', alpha=0.5, label='Bollinger Bands (±2σ, 12w)')

# Highlight spike weeks
spike_dates = wm_ts[wm_spikes].index
for sd in spike_dates:
    ax1.axvspan(sd - pd.Timedelta(days=3), sd + pd.Timedelta(days=3), color='#d73027', alpha=0.3)
    ax2.axvspan(sd - pd.Timedelta(days=3), sd + pd.Timedelta(days=3), color='#d73027', alpha=0.3)

ax1.scatter(spike_dates, wm_ts[spike_dates], color='#d73027', s=50, zorder=5, label='Spike Week (>5% 4w rise)')
ax1.set_title('Primary Target: Retail Price of Well-Milled Rice (Local) in Metro Manila (Aug 2023 – Oct 2026)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price (₱ / kg)', fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', frameon=True)
ax1.set_ylim(40, 56)

# Annotate spike clusters
ax1.annotate('Cluster 1: Post-EO 39 Lift Spike\n(Nov–Dec 2023)', 
             xy=(pd.Timestamp('2023-11-20'), 49.3), 
             xytext=(pd.Timestamp('2023-09-01'), 54.0),
             arrowprops=dict(facecolor='#d73027', shrink=0.08, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee08b', alpha=0.9))

ax1.annotate('Cluster 2: Lean Season Transition Surge\n(Apr 2026, +14.8% 4w jump)', 
             xy=(pd.Timestamp('2026-04-20'), 50.45), 
             xytext=(pd.Timestamp('2025-08-01'), 53.5),
             arrowprops=dict(facecolor='#d73027', shrink=0.08, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee08b', alpha=0.9))

# Bottom panel: 4-week percentage change
ax2.plot(wm_pct4.index, wm_pct4, color='#313695', lw=1.8, label='4-Week % Change')
ax2.axhline(5.0, color='#d73027', linestyle='--', lw=1.5, label='Spike Threshold (+5.0%)')
ax2.axhline(0.0, color='gray', linestyle=':', lw=1)
ax2.scatter(spike_dates, wm_pct4[spike_dates], color='#d73027', s=45, zorder=5)

ax2.set_title('4-Week Rate of Change & Spike Trigger Signal', fontsize=12, fontweight='bold')
ax2.set_ylabel('4-Week Change (%)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Date (Weekly Frequency)', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', frameon=True)
ax2.set_ylim(-10, 20)

plt.tight_layout()
plt.savefig('images/fig2_well_milled_rice_spike_detection.png', dpi=300)
plt.close()
print('Fig 2 saved.')

# FIG 3: Univariate Analysis: Distributions & Category Volatility
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2)

# 3A: Histogram & KDE of Well Milled Rice
ax_hist = fig.add_subplot(gs[0, 0])
sns.histplot(wm_ts, kde=True, ax=ax_hist, color='#2ca02c', bins=20, stat='density', alpha=0.6)
ax_hist.axvline(wm_ts.mean(), color='#d62728', linestyle='--', lw=2, label=f'Mean: ₱{wm_ts.mean():.2f}')
ax_hist.axvline(wm_ts.median(), color='#1f77b4', linestyle='-', lw=2, label=f'Median: ₱{wm_ts.median():.2f}')
ax_hist.set_title('A. Price Distribution: Well-Milled Rice (Local)', fontsize=12, fontweight='bold')
ax_hist.set_xlabel('Price (₱/kg)')
ax_hist.set_ylabel('Density')
ax_hist.legend(loc='upper right')

# 3B: Boxplot of Rice Varieties
ax_box = fig.add_subplot(gs[0, 1])
rice_items = [
    'Rice, Local - Regular Milled',
    'Rice, Local - Well Milled',
    'Rice, Local - Premium',
    'Rice, Local - Special',
    'Rice, Imported - Well Milled'
]
rice_df = df[df['commodity_std'].isin(rice_items)].copy()
rice_order = ['Rice, Local - Regular Milled', 'Rice, Local - Well Milled', 'Rice, Imported - Well Milled', 'Rice, Local - Premium', 'Rice, Local - Special']
labels = ['Regular Milled', 'Well Milled (Local)', 'Well Milled (Imp)', 'Premium', 'Special']
sns.boxplot(data=rice_df, x='commodity_std', y='price_num', order=rice_order, ax=ax_box, palette='YlOrBr')
ax_box.set_xticklabels(labels, rotation=15, ha='right')
ax_box.set_title('B. Price Comparison Across Rice Grades', fontsize=12, fontweight='bold')
ax_box.set_xlabel('')
ax_box.set_ylabel('Price (₱/kg)')

# 3C: Category Volatility (Coefficient of Variation: std / mean * 100)
ax_cv = fig.add_subplot(gs[1, :])
cv_data = []
for cat in df['category'].unique():
    cat_prices = df[(df['category'] == cat) & (df['price_num'].notna())]['price_num']
    if len(cat_prices) > 50:
        cv = (cat_prices.std() / cat_prices.mean()) * 100
        cv_data.append({'Category': cat, 'CV': cv, 'Mean': cat_prices.mean(), 'Std': cat_prices.std()})
cv_df = pd.DataFrame(cv_data).sort_values('CV', ascending=False)

sns.barplot(data=cv_df, x='CV', y='Category', ax=ax_cv, palette='viridis')
ax_cv.set_title('C. Commodity Category Volatility Ranking (Coefficient of Variation: σ / μ × 100%)', fontsize=12, fontweight='bold')
ax_cv.set_xlabel('Coefficient of Variation (CV %)', fontsize=11, fontweight='bold')
ax_cv.set_ylabel('')
for i, r in enumerate(cv_df.itertuples()):
    ax_cv.text(r.CV + 0.5, i, f'{r.CV:.1f}% (μ=₱{r.Mean:.1f})', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('images/fig3_univariate_distributions.png', dpi=300)
plt.close()
print('Fig 3 saved.')

# FIG 4: Bivariate: Rice Varieties Comparison & Margin Spreads
fig, (ax_ts, ax_spr) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

rice_piv = df[df['commodity_std'].isin(rice_items)].pivot_table(index='week_start', columns='commodity_std', values='price_num').reindex(full_idx).interpolate(method='linear')

colors_map = {
    'Rice, Local - Special': '#756bb1',
    'Rice, Local - Premium': '#2ca02c',
    'Rice, Local - Well Milled': '#1b7837',
    'Rice, Imported - Well Milled': '#e6550d',
    'Rice, Local - Regular Milled': '#3182bd'
}

for col in rice_order:
    if col in rice_piv.columns:
        ax_ts.plot(rice_piv.index, rice_piv[col], label=col.replace('Rice, ', ''), color=colors_map[col], lw=2 if 'Well Milled' in col else 1.5)

ax_ts.set_title('A. Weekly Retail Price Trajectories of Key Rice Commodities (NCR)', fontsize=13, fontweight='bold')
ax_ts.set_ylabel('Price (₱ / kg)', fontsize=11, fontweight='bold')
ax_ts.legend(loc='upper left', frameon=True)

# Spread panel
spread_wm_rm = rice_piv['Rice, Local - Well Milled'] - rice_piv['Rice, Local - Regular Milled']
spread_wm_imp = rice_piv['Rice, Local - Well Milled'] - rice_piv['Rice, Imported - Well Milled']

ax_spr.plot(rice_piv.index, spread_wm_rm, label='Local Well Milled - Regular Milled Spread', color='#3182bd', lw=1.5)
ax_spr.plot(rice_piv.index, spread_wm_imp, label='Local Well Milled - Imported Well Milled Spread', color='#e6550d', lw=1.5, ls='--')
ax_spr.axhline(0, color='gray', linestyle=':')
ax_spr.set_title('B. Price Margins / Spreads Relative to Well-Milled Rice', fontsize=12, fontweight='bold')
ax_spr.set_ylabel('Spread (₱ / kg)', fontsize=11, fontweight='bold')
ax_spr.set_xlabel('Date', fontsize=11, fontweight='bold')
ax_spr.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('images/fig4_bivariate_time_series_comparison.png', dpi=300)
plt.close()
print('Fig 4 saved.')

# FIG 5: Seasonality Analysis
fig, (ax_m, ax_q) = plt.subplots(1, 2, figsize=(16, 5))

# Monthly seasonality
wm_seasonal = pd.DataFrame({'price': wm_ts})
wm_seasonal['month'] = wm_seasonal.index.month
wm_seasonal['quarter'] = wm_seasonal.index.quarter
wm_seasonal['month_name'] = wm_seasonal.index.strftime('%b')
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sns.boxplot(data=wm_seasonal, x='month_name', y='price', order=month_order, ax=ax_m, palette='crest')
ax_m.set_title('A. Monthly Price Variation (Well-Milled Rice)', fontsize=12, fontweight='bold')
ax_m.set_xlabel('Month', fontsize=10, fontweight='bold')
ax_m.set_ylabel('Price (₱ / kg)', fontsize=10, fontweight='bold')

# Quarterly seasonality
sns.boxplot(data=wm_seasonal, x='quarter', y='price', ax=ax_q, palette='magma')
ax_q.set_xticklabels(['Q1 (Jan–Mar)', 'Q2 (Apr–Jun)', 'Q3 (Jul–Sep / Lean)', 'Q4 (Oct–Dec / Harvest)'])
ax_q.set_title('B. Quarterly Price Behavior: Lean vs. Harvest Seasons', fontsize=12, fontweight='bold')
ax_q.set_xlabel('Quarter', fontsize=10, fontweight='bold')
ax_q.set_ylabel('Price (₱ / kg)', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('images/fig5_seasonality_analysis.png', dpi=300)
plt.close()
print('Fig 5 saved.')

# FIG 6: Cross-Commodity Normalized Index Dynamics
commodities_rep = {
    'Rice, Local - Well Milled': 'Well Milled Rice',
    'Bangus (Medium)': 'Bangus (Fish)',
    'Tilapia': 'Tilapia (Fish)',
    'Galunggong': 'Galunggong (Fish)',
    'Ampalaya': 'Ampalaya (Veg)',
    'Tomato': 'Tomato (Veg)',
    'Cabbage Scorpio': 'Cabbage (Veg)',
    'Carrots': 'Carrots (Veg)'
}

piv_rep = df[df['commodity_std'].isin(commodities_rep.keys())].pivot_table(index='week_start', columns='commodity_std', values='price_num').reindex(full_idx).interpolate(method='linear')
piv_rep.rename(columns=commodities_rep, inplace=True)

# Normalize to index = 100 at start
norm_idx = (piv_rep / piv_rep.iloc[0]) * 100

plt.figure(figsize=(15, 7))
plt.plot(norm_idx.index, norm_idx['Well Milled Rice'], color='#00441b', lw=3, label='★ Well Milled Rice (Primary Target)')
for col in norm_idx.columns:
    if col != 'Well Milled Rice':
        alpha = 0.8 if 'Fish' in col else 0.5
        ls = '-' if 'Fish' in col else '--'
        plt.plot(norm_idx.index, norm_idx[col], label=col, alpha=alpha, ls=ls, lw=1.5)

plt.axhline(100, color='gray', linestyle=':', lw=1)
plt.title('Normalized Price Index Comparison: Rice vs. Fish vs. Vegetables (Base = Aug 2023 = 100)', fontsize=14, fontweight='bold')
plt.ylabel('Normalized Price Index (Base = 100)', fontsize=11, fontweight='bold')
plt.xlabel('Date', fontsize=11, fontweight='bold')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('images/fig6_cross_commodity_dynamics.png', dpi=300)
plt.close()
print('Fig 6 saved.')

# FIG 7: Multivariate Correlation Heatmaps
fig, (ax_corr_lvl, ax_corr_ret) = plt.subplots(1, 2, figsize=(18, 7))

# 14 commodities
commodities_all = [
    'Rice, Local - Well Milled',
    'Rice, Local - Regular Milled',
    'Rice, Local - Premium',
    'Rice, Imported - Well Milled',
    'Bangus (Medium)',
    'Tilapia',
    'Galunggong',
    'Ampalaya',
    'Eggplant',
    'Tomato',
    'Cabbage Scorpio',
    'Carrots',
    'Red Onion',
    'Pork Belly'
]
short_names = [c.replace('Rice, Local - ', 'Rice L-').replace('Rice, Imported - ', 'Rice I-') for c in commodities_all]

sub_all = df[df['commodity_std'].isin(commodities_all)].pivot_table(index='week_start', columns='commodity_std', values='price_num').reindex(full_idx).interpolate(method='linear')[commodities_all]
sub_all.columns = short_names

# Correlation of price levels
corr_lvl = sub_all.corr()
mask = np.triu(np.ones_like(corr_lvl, dtype=bool))
sns.heatmap(corr_lvl, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax_corr_lvl, cbar_kws={'label': 'Pearson r'})
ax_corr_lvl.set_title('A. Price Level Correlation (Multi-Year Trends)', fontsize=12, fontweight='bold')

# Correlation of 4-week percentage changes
sub_pct4 = sub_all.pct_change(4).dropna()
corr_ret = sub_pct4.corr()
sns.heatmap(corr_ret, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax_corr_ret, cbar_kws={'label': 'Pearson r'})
ax_corr_ret.set_title('B. 4-Week % Change Correlation (Short-Term Shock Co-movement)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('images/fig7_multivariate_correlation_heatmaps.png', dpi=300)
plt.close()
print('Fig 7 saved.')

# FIG 8: Lead-Lag Cross Correlation
lags = range(-8, 9)
ccf_results = {}
wm_series = sub_all['Rice L-Well Milled']
for col in ['Rice L-Regular Milled', 'Rice I-Well Milled', 'Bangus (Medium)', 'Tilapia', 'Ampalaya', 'Tomato', 'Cabbage Scorpio']:
    s = sub_all[col]
    corrs = [wm_series.corr(s.shift(-lag)) for lag in lags]
    ccf_results[col] = corrs

plt.figure(figsize=(12, 6))
for col, corrs in ccf_results.items():
    lw = 2.5 if 'Rice' in col else 1.5
    plt.plot(list(lags), corrs, label=col, marker='o', markersize=4, lw=lw)

plt.axvline(0, color='gray', linestyle='--', lw=1)
plt.axhline(0, color='gray', linestyle=':', lw=1)
plt.title('Lead-Lag Cross-Correlation with Well-Milled Rice (Lags: -8 to +8 Weeks)', fontsize=13, fontweight='bold')
plt.xlabel('Lag k in Weeks (Negative: Commodity Leads Rice | Positive: Rice Leads Commodity)', fontsize=11, fontweight='bold')
plt.ylabel('Correlation r(k)', fontsize=11, fontweight='bold')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('images/fig8_lead_lag_cross_correlation.png', dpi=300)
plt.close()
print('Fig 8 saved.')
print('All figures generated successfully.')
