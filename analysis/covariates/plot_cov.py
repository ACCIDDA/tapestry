import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = pd.read_csv('covariate_results.csv')
SRC = [('claims_outpatient','Outpatient claims','#2F6EA8'),
       ('claims_inpatient','Inpatient claims','#7A7A7A'),
       ('pophive_ed','PopHIVE ED','#B4472A'),
       ('wastewater','Wastewater index','#4F8A54')]
PATH = [('flu','Influenza A'),('covid','SARS-CoV-2'),('rsv','RSV')]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
for ax,(pk,plabel) in zip(axes, PATH):
    for key,label,c in SRC:
        d = r[(r.source==key)&(r.pathogen==pk)].sort_values('h')
        if d.empty: continue
        ax.plot(d['h'], d['gain'], 'o-', color=c, lw=1.8, ms=5, label=label)
    ax.axhline(0, color='0.4', lw=0.9)
    ax.set_title(plabel, fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel('forecast horizon (weeks)')
    ax.set_xticks([1,2,3,4])
    ax.grid(True, color='0.92', lw=0.6)
    for s in ('top','right'): ax.spines[s].set_visible(False)
axes[0].set_ylabel('gain in adjusted R² over the NHSN-only baseline')
axes[0].legend(frameon=False, fontsize=9.5, loc='upper left')
fig.suptitle('Incremental value of Delphi covariates at a Wednesday forecast origin\n'
             'NHSN autoregressive baseline rebuilt from real vintages; adjusted R², median over states',
             x=0.006, ha='left', fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.88])
fig.savefig('covariates.png', dpi=140)
print('wrote covariates.png')
