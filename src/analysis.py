"""VK-LSVD: collaborative vs content signal sensitivity. Self-contained, functional."""
import os, json, numpy as np, polars as pl, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
RNG = np.random.default_rng(0); OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
K = 10; DIMS = [8, 16, 32, 64]; ALPHAS = [0.1, 0.5, 1.0]; np.seterr(all="ignore")

def engaged():
    dur = pl.when(pl.col("duration") == 0).then(1).otherwise(pl.col("duration"))
    return (pl.col("like") | pl.col("share") | pl.col("bookmark")
            | pl.col("click_on_author") | (pl.col("timespent")/dur >= 0.7))

def load():
    im = pl.read_parquet("items_metadata.parquet")
    tr = pl.read_parquet("train.parquet").join(im.select(["item_id","duration"]), on="item_id", how="left")
    va = pl.read_parquet("val.parquet").join(im.select(["item_id","duration"]), on="item_id", how="left")
    return tr.with_columns(engaged().alias("eng")), va.with_columns(engaged().alias("eng")), im, np.load("item_embeddings.npz")

def build(tr, va, im, emb):
    users = tr["user_id"].unique().sort().to_list(); items = im["item_id"].unique().sort().to_list()
    uix = {u:i for i,u in enumerate(users)}; iix = {v:i for i,v in enumerate(items)}
    nU, nI = len(users), len(items)
    trp = tr.filter(pl.col("eng"))
    ru = np.array([uix[u] for u in trp["user_id"]]); ri = np.array([iix[v] for v in trp["item_id"]])
    R = sp.csr_matrix((np.ones(len(ru),np.float32),(ru,ri)), shape=(nU,nI)); R.sum_duplicates(); R.data[:] = 1.0
    vp = va.filter(pl.col("eng")).filter(pl.col("item_id").is_in(items)); val_pos = {}
    for u,v in zip(vp["user_id"].to_list(), vp["item_id"].to_list()): val_pos.setdefault(uix[u],set()).add(iix[v])
    val_users = np.array(sorted(val_pos))
    eid = {int(x):k for k,x in enumerate(emb["item_id"])}; E = np.zeros((nI, emb["embedding"].shape[1]), np.float32)
    for v,j in iix.items(): E[j] = emb["embedding"][eid[v]]
    return dict(nU=nU, nI=nI, R=R, val_pos=val_pos, val_users=val_users, E=E)

def als_scores(D, factors=64, reg=0.05, iters=25):
    m = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters, random_state=0, use_gpu=False)
    m.fit(D["R"], show_progress=False)
    return (m.user_factors[D["val_users"]] @ m.item_factors.T).astype(np.float32)

def itemknn_scores(D, k=200):
    m = CosineRecommender(K=k); m.fit(D["R"], show_progress=False)
    return np.asarray((D["R"][D["val_users"]] @ m.similarity).todense(), np.float32)

def userknn_scores(D, k=100):
    R = D["R"]; nrm = np.sqrt(np.asarray(R.multiply(R).sum(1)).ravel()); nrm[nrm==0] = 1
    Rn = R.multiply(1/nrm[:,None]).tocsr(); sim = (Rn[D["val_users"]] @ Rn.T).toarray()
    for r,u in enumerate(D["val_users"]): sim[r,u] = 0.0
    if k < sim.shape[1]:
        cut = np.partition(sim,-k,axis=1)[:,-k][:,None]; sim[sim<cut] = 0
    return (sim @ R).astype(np.float32)

def content_scores(D, dim):
    E = D["E"][:,:dim]; prof = D["R"][D["val_users"]] @ E
    cnt = np.asarray(D["R"][D["val_users"]].sum(1)).ravel(); cnt[cnt==0] = 1
    return ((prof/cnt[:,None]) @ E.T).astype(np.float32)

def zrows(S):
    mu = S.mean(1,keepdims=True); sd = S.std(1,keepdims=True); sd[sd==0] = 1; return (S-mu)/sd
def fuse(cf, content, alpha): return zrows(cf) + alpha*zrows(content)

def per_user_metrics(S, D):
    vu = D["val_users"]; S = S.copy(); seen = D["R"][vu].tolil().rows
    for r,cols in enumerate(seen):
        if cols: S[r,cols] = -np.inf
    order = np.argsort(-S,axis=1)[:,:K]; hit1=[]; hit10=[]; ndcg=[]; top=set()
    for r in range(len(vu)):
        pos = D["val_pos"][vu[r]]; rank = order[r]; top.update(rank.tolist())
        h = [1 if j in pos else 0 for j in rank]; hit1.append(h[0]); hit10.append(1 if sum(h) else 0)
        idcg = sum(1/np.log2(i+2) for i in range(min(len(pos),K)))
        ndcg.append(sum(h[i]/np.log2(i+2) for i in range(K))/idcg if idcg else 0.0)
    return dict(hit1=np.array(hit1), hit10=np.array(hit10), ndcg=np.array(ndcg), coverage=len(top)/D["nI"])

def agg(m, mask):
    return dict(HitRate_1=float(np.nanmean(m["hit1"][mask])), HitRate_10=float(np.nanmean(m["hit10"][mask])),
                NDCG_10=float(np.nanmean(m["ndcg"][mask])), Coverage_10=float(m["coverage"]))

def bootstrap_diff(a, b, mask, B=1000):
    a=a[mask]; b=b[mask]; n=len(a); idx=RNG.integers(0,n,(B,n)); diffs=b[idx].mean(1)-a[idx].mean(1)
    lo,hi = np.percentile(diffs,[2.5,97.5]); p = 2*min((diffs<=0).mean(),(diffs>=0).mean())
    return float(b.mean()-a.mean()), float(lo), float(hi), float(min(p,1.0))

# ============================ RUN ============================
print("loading..."); tr,va,im,emb = load(); D = build(tr,va,im,emb)
stats = {"users":D["nU"], "items":D["nI"], "train_pos":int(D["R"].nnz), "val_users":int(len(D["val_users"]))}
print(stats); json.dump(stats, open(f"{OUT}/_stats.json","w"))
vu = D["val_users"].copy(); RNG.shuffle(vu); cut=int(0.3*len(vu)); tune_u=set(vu[:cut].tolist())
tune_mask = np.array([u in tune_u for u in D["val_users"]]); test_mask = ~tune_mask

print("collaborative models..."); S_als=als_scores(D); S_ik=itemknn_scores(D); S_uk=userknn_scores(D)

grid=[]
for a in ALPHAS:
    mt = per_user_metrics(fuse(S_als, content_scores(D,64), a), D)
    grid.append(dict(alpha=a, tune_NDCG=agg(mt,tune_mask)["NDCG_10"], test_NDCG=agg(mt,test_mask)["NDCG_10"]))
pl.DataFrame(grid).write_csv(f"{OUT}/alpha_grid.csv")
best_alpha = max(grid, key=lambda r:r["tune_NDCG"])["alpha"]; print("best_alpha", best_alpha)

sweep=[]; cc={}
for d in DIMS:
    cc[d] = content_scores(D,d); mt = per_user_metrics(fuse(S_als,cc[d],best_alpha), D)
    a_=agg(mt,test_mask); a_["dim"]=d; sweep.append(a_)
pl.DataFrame(sweep).select(["dim","HitRate_1","HitRate_10","NDCG_10","Coverage_10"]).write_csv(f"{OUT}/dim_sweep.csv")
best_dim = max(sweep, key=lambda r:r["NDCG_10"])["dim"]; print("best_dim", best_dim)

S_fuse = fuse(S_als, cc[best_dim], best_alpha); S_con = content_scores(D,best_dim)
fuse_name = f"ALS+Content (a={best_alpha}, d={best_dim})"
models = {"ALS (CF only)":S_als, "ItemKNN":S_ik, "UserKNN":S_uk, "Content only":S_con, fuse_name:S_fuse}
rows={}; peruser={}
main=[]
for nm,S in models.items():
    mt=per_user_metrics(S,D); peruser[nm]=mt; r=agg(mt,test_mask)
    main.append({"model":nm, **{k:round(v,5) for k,v in r.items()}})
pl.DataFrame(main).write_csv(f"{OUT}/metrics_main.csv"); print("MAIN:\n", pl.DataFrame(main))

# cold-start: hold out 15% of items from ALS training; rank within cold pool
RC = np.random.default_rng(7); cold_hold = np.sort(RC.choice(D["nI"], int(0.15*D["nI"]), replace=False))
cold_set = set(cold_hold.tolist()); Rw = D["R"].tolil(); Rw[:,cold_hold]=0; Rw=Rw.tocsr(); Rw.eliminate_zeros()
mc = AlternatingLeastSquares(factors=64,regularization=0.05,iterations=25,random_state=0,use_gpu=False); mc.fit(Rw,show_progress=False)
als_c = (mc.user_factors[D["val_users"]] @ mc.item_factors[cold_hold].T).astype(np.float32)
Ec = D["E"][cold_hold,:best_dim]; prof = D["R"][D["val_users"]] @ D["E"][:,:best_dim]
cnt = np.asarray(D["R"][D["val_users"]].sum(1)).ravel(); cnt[cnt==0]=1
con_c = ((prof/cnt[:,None]) @ Ec.T).astype(np.float32); fus_c = zrows(als_c)+best_alpha*zrows(con_c)
cix = {c:k for k,c in enumerate(cold_hold)}; seen = D["R"][D["val_users"]].tolil().rows
def cold_eval(S):
    hr=[]; nd=[]; used=[]
    for r in range(len(D["val_users"])):
        posc=[cix[j] for j in (D["val_pos"][D["val_users"][r]] & cold_set)]
        if not posc: continue
        s=S[r].copy()
        for j in seen[r]:
            if j in cix: s[cix[j]]=-np.inf
        order=np.argsort(-s)[:K]; h=[1 if j in posc else 0 for j in order]
        idcg=sum(1/np.log2(i+2) for i in range(min(len(posc),K)))
        hr.append(1 if sum(h) else 0); nd.append(sum(h[i]/np.log2(i+2) for i in range(K))/idcg); used.append(r)
    used=np.array(used); m=test_mask[used]
    return float(np.array(hr)[m].mean()), float(np.array(nd)[m].mean()), int(m.sum())
cold=[]
for nm,S in [("ALS (CF only)",als_c),("Content only",con_c),("ALS+Content",fus_c)]:
    hr,nd,n = cold_eval(S); cold.append({"model":nm,"HitRate_10":round(hr,5),"NDCG_10":round(nd,5),"n_users":n})
pl.DataFrame(cold).write_csv(f"{OUT}/coldstart.csv"); print("COLD:\n", pl.DataFrame(cold))

base = peruser["ALS (CF only)"]["ndcg"]; boot=[]
for nm in ["ItemKNN","UserKNN","Content only",fuse_name]:
    d_,lo,hi,p = bootstrap_diff(base, peruser[nm]["ndcg"], test_mask)
    boot.append(dict(comparison=f"{nm} vs ALS", delta_NDCG=round(d_,5), ci_lo=round(lo,5), ci_hi=round(hi,5),
                     p_value=round(p,4), significant=bool(p<0.05)))
pl.DataFrame(boot).write_csv(f"{OUT}/bootstrap.csv"); print("BOOT:\n", pl.DataFrame(boot))

# figures
C={"cf":"#2E5EAA","fuse":"#C1440E","content":"#3C8D40"}
plt.figure(figsize=(7,4))
for nm,S,c in [("ALS",S_als,C["cf"]),("Content",S_con,C["content"]),("ALS+Content",S_fuse,C["fuse"])]:
    v=S[test_mask].ravel(); v=v[np.isfinite(v)]; plt.hist(v[RNG.integers(0,len(v),60000)],bins=80,alpha=0.5,density=True,label=nm,color=c)
plt.xlabel("predicted score"); plt.ylabel("density"); plt.legend(); plt.title("Distribution of predicted scores")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_score_dist.png",dpi=130); plt.close()

als_ndcg = agg(peruser["ALS (CF only)"],test_mask)["NDCG_10"]
plt.figure(figsize=(6,4)); plt.plot([r["alpha"] for r in grid],[r["test_NDCG"] for r in grid],"o-",color=C["fuse"])
plt.axhline(als_ndcg,ls="--",color=C["cf"],label="ALS baseline"); plt.xlabel("fusion weight alpha"); plt.ylabel("NDCG@10 (test)")
plt.legend(); plt.title("Late-fusion weight vs NDCG@10"); plt.tight_layout(); plt.savefig(f"{OUT}/fig_alpha.png",dpi=130); plt.close()

plt.figure(figsize=(6,4)); plt.plot([r["dim"] for r in sweep],[r["NDCG_10"] for r in sweep],"s-",color=C["content"])
plt.axhline(als_ndcg,ls="--",color=C["cf"],label="ALS baseline"); plt.xlabel("content embedding dimensionality"); plt.ylabel("NDCG@10 (test)")
plt.xticks(DIMS); plt.legend(); plt.title("Content signal strength (embedding dim)"); plt.tight_layout(); plt.savefig(f"{OUT}/fig_dim.png",dpi=130); plt.close()

axes=["HitRate_1","HitRate_10","NDCG_10","Coverage_10"]; labels=["HitRate@1","HitRate@10","NDCG@10","Coverage@10"]
ang=np.linspace(0,2*np.pi,len(axes),endpoint=False).tolist(); ang+=ang[:1]
fig,ax=plt.subplots(figsize=(6.2,6.2),subplot_kw=dict(polar=True)); cols=[C["cf"],"#888","#B47ACy","#3C8D40",C["fuse"]]
cols=[C["cf"],"#888888","#B47AC2",C["content"],C["fuse"]]
mx={k:max(agg(peruser[nm],test_mask)[k] for nm in models) for k in axes}
for (nm),col in zip(models,cols):
    a_=agg(peruser[nm],test_mask); norm=[a_[axes[i]]/mx[axes[i]] if mx[axes[i]] else 0 for i in range(len(axes))]; norm+=norm[:1]
    ax.plot(ang,norm,"-",lw=2,label=nm,color=col); ax.fill(ang,norm,alpha=0.07,color=col)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels); ax.set_yticklabels([]); ax.set_title("Model comparison (normalized)",pad=20)
ax.legend(loc="upper right",bbox_to_anchor=(1.4,1.12),fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/fig_radar.png",dpi=130,bbox_inches="tight"); plt.close()

plt.figure(figsize=(6,4)); nm=[r["model"] for r in cold]
plt.bar(nm,[r["HitRate_10"] for r in cold],color=[C["cf"],C["content"],C["fuse"]])
plt.ylabel("HitRate@10 (cold pool)"); plt.title("Cold-start: ranking 15% held-out items"); plt.tight_layout(); plt.savefig(f"{OUT}/fig_cold.png",dpi=130); plt.close()

json.dump({"best_alpha":best_alpha,"best_dim":best_dim,"n_test":int(test_mask.sum()),"n_tune":int(tune_mask.sum()),
           "als_ndcg":als_ndcg}, open(f"{OUT}/_choices.json","w"))
print("DONE:", sorted(os.listdir(OUT)))
