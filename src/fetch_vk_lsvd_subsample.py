"""Скачивание подвыборки VK-LSVD и сборка компактного бандла (curl в обход HEAD-эндпоинта HF)."""
import os, subprocess, polars as pl, numpy as np
REPO, SUBSAMPLE, N_TRAIN_WEEKS = "deepvk/VK-LSVD", "up0.001_ip0.001", 25
N_USERS, MAX_INT_PER_USER, EMB_DIM, OUT = 1500, 400, 64, "vk_lsvd_mini"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"; os.makedirs(OUT, exist_ok=True)
def fetch(rel):
    out=f"VK-LSVD/{rel}"; os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out)>0: return out
    subprocess.run(["curl","-L","--fail","--retry","20","--retry-delay","5","--retry-all-errors",
                    "-C","-","-o",out,f"{BASE}/{rel}?download=true"], check=True); return out
tf=[f"subsamples/{SUBSAMPLE}/train/week_{i:02}.parquet" for i in range(N_TRAIN_WEEKS)]
vf=f"subsamples/{SUBSAMPLE}/validation/week_{N_TRAIN_WEEKS:02}.parquet"
mf=["metadata/users_metadata.parquet","metadata/items_metadata.parquet","metadata/item_embeddings.npz"]
for f in tf+[vf]+mf: fetch(f)
tr=pl.concat([pl.scan_parquet(f"VK-LSVD/{f}") for f in tf]).collect(engine="streaming"); va=pl.read_parquet(f"VK-LSVD/{vf}")
ku=tr.select("user_id").unique().sort("user_id").head(N_USERS); tr,va=tr.join(ku,on="user_id"),va.join(ku,on="user_id")
tr=(tr.with_columns(pl.int_range(pl.len()).over("user_id").alias("i"),pl.len().over("user_id").alias("c"))
      .filter(pl.col("i")>=pl.col("c")-MAX_INT_PER_USER).drop(["i","c"]))
ki=tr.select("item_id").unique(); e=np.load("VK-LSVD/metadata/item_embeddings.npz"); ids,em=e["item_id"],e["embedding"]
m=np.isin(ids,ki.to_numpy().ravel()); ids,em=ids[m],em[m][:,:EMB_DIM]
tr.write_parquet(f"{OUT}/train.parquet",compression="zstd"); va.write_parquet(f"{OUT}/val.parquet",compression="zstd")
pl.read_parquet("VK-LSVD/metadata/items_metadata.parquet").join(ki,on="item_id").write_parquet(f"{OUT}/items_metadata.parquet",compression="zstd")
pl.read_parquet("VK-LSVD/metadata/users_metadata.parquet").join(ku,on="user_id").write_parquet(f"{OUT}/users_metadata.parquet",compression="zstd")
np.savez_compressed(f"{OUT}/item_embeddings.npz",item_id=ids,embedding=em)
for fn in sorted(os.listdir(OUT)): print(fn, round(os.path.getsize(f"{OUT}/{fn}")/1e6,2),"MB")
