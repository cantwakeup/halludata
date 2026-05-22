import argparse, csv, math
from pathlib import Path
from collections import defaultdict

def f(x, d=0.0):
    try: return float(x) if x not in ("", None) else d
    except Exception: return d

def i(x, d=0):
    try: return int(float(x)) if x not in ("", None) else d
    except Exception: return d

def md_table(headers, rows):
    out=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]
    for r in rows:
        vals=[]
        for h in headers:
            v=r.get(h,"")
            vals.append(f"{v:.2f}" if isinstance(v,float) else str(v))
        out.append("| "+" | ".join(vals)+" |")
    return "\n".join(out)

ap=argparse.ArgumentParser()
ap.add_argument("--summary", required=True)
ap.add_argument("--output-dir", required=True)
args=ap.parse_args()

summary=Path(args.summary)
out=Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)

rows=list(csv.DictReader(open(summary, newline="", encoding="utf-8")))
rows=sorted(rows, key=lambda r:(r["Dataset"],r["Setting"],0 if r["Method"]=="Regular" else 1,f(r["Alpha"],-1)))

base={(r["Dataset"],r["Setting"]):r for r in rows if r["Method"]=="Regular"}
cat=[r for r in rows if r["Method"]=="CatExpert"]

complete=[]
bad=[]
for r in cat:
    b=base[(r["Dataset"],r["Setting"])]
    if i(r["N"])==i(b["N"]): complete.append(r)
    else: bad.append(r)

delta=[]
for r in complete:
    b=base[(r["Dataset"],r["Setting"])]
    rr=dict(r)
    for m in ["Accuracy","Precision","Recall","F1 Score","Yes Rate"]:
        rr["Delta "+m]=f(r[m])-f(b[m])
    for m in ["FP","FN","TP","TN"]:
        rr["Delta "+m]=i(r[m])-i(b[m])
    delta.append(rr)

best=[]
groups=defaultdict(list)
for r in delta:
    groups[(r["Dataset"],r["Setting"])].append(r)
for k,g in sorted(groups.items()):
    r=max(g, key=lambda x:(f(x["F1 Score"]),f(x["Accuracy"])))
    best.append({
        "Dataset":k[0],"Setting":k[1],"Best Alpha":r["Alpha"],
        "Best Accuracy":f(r["Accuracy"]),"Delta Accuracy":r["Delta Accuracy"],
        "Best F1 Score":f(r["F1 Score"]),"Delta F1 Score":r["Delta F1 Score"],
        "Best FP":r["FP"],"Delta FP":r["Delta FP"],
        "Best FN":r["FN"],"Delta FN":r["Delta FN"],
        "Best Yes Rate":f(r["Yes Rate"]),"Delta Yes Rate":r["Delta Yes Rate"],
    })

def write_csv(name, rs, headers):
    with open(out/name,"w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=headers); w.writeheader()
        for r in rs: w.writerow({x:r.get(x,"") for x in headers})

full_h=["Dataset","Setting","Method","Alpha","N","Accuracy","Precision","Recall","F1 Score","Yes Rate","TP","TN","FP","FN","Invalid"]
delta_h=["Dataset","Setting","Alpha","N","Accuracy","Delta Accuracy","F1 Score","Delta F1 Score","Precision","Delta Precision","Recall","Delta Recall","Yes Rate","Delta Yes Rate","FP","Delta FP","FN","Delta FN"]
best_h=["Dataset","Setting","Best Alpha","Best Accuracy","Delta Accuracy","Best F1 Score","Delta F1 Score","Best FP","Delta FP","Best FN","Delta FN","Best Yes Rate","Delta Yes Rate"]

write_csv("full_summary_table.csv",rows,full_h)
write_csv("cat_alpha_deltas.csv",delta,delta_h)
write_csv("best_by_f1.csv",best,best_h)
if bad: write_csv("incomplete_rows.csv",bad,full_h)

try:
    import matplotlib.pyplot as plt
    metrics=["Accuracy","F1 Score","Precision","Recall","Yes Rate","FP","FN"]
    for metric in metrics:
        keys=sorted(groups)
        cols=3; nrows=math.ceil(len(keys)/cols)
        fig,axs=plt.subplots(nrows,cols,figsize=(cols*5.2,nrows*3.6),squeeze=False)
        for ax in axs.ravel(): ax.set_visible(False)
        for ax,k in zip(axs.ravel(),keys):
            ax.set_visible(True)
            g=sorted(groups[k],key=lambda r:f(r["Alpha"]))
            ax.plot([f(r["Alpha"]) for r in g],[f(r[metric]) for r in g],marker="o",label="CatExpert")
            ax.axhline(f(base[k][metric]),ls="--",color="gray",label="Regular")
            ax.set_title(f"{k[0]} {k[1]}")
            ax.set_xlabel("alpha"); ax.set_ylabel(metric); ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out/f"alpha_curve_{metric.lower().replace(' ','_')}.png",dpi=160)
        plt.close(fig)
except Exception as e:
    print("plot skipped:",e)

report=[
"# POPE CatExpert Alpha Tables",
"",
"## Best CatExpert By F1",
md_table(best_h,best),
"",
"## Full Summary Table",
md_table(full_h,rows),
]
if bad:
    report += ["","## Incomplete Rows",md_table(full_h,bad)]
(out/"ALPHA_CURVES.md").write_text("\n".join(report)+"\n",encoding="utf-8")
print("wrote",out)
