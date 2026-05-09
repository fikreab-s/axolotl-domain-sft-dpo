"""Generate preference pairs for DPO training."""
import json, random, numpy as np, argparse
from pathlib import Path
random.seed(42); np.random.seed(42)
BRANDS = ["Cardivex","Immunolex","OncoPrime","NeuraStar","RespiClear"]

def gen_pair():
    brand = random.choice(BRANDS)
    q = f"What is the ROI outlook for {brand}?"
    good = f"Based on the MMM analysis, {brand} shows a {random.uniform(2,5):.1f}x ROI with strong adstock carryover. Recommend maintaining current spend levels with a {random.randint(5,15)}% shift to digital."
    bad = f"{brand} ROI is {'good' if random.random()>0.5 else 'okay'}. Keep doing what we are doing."
    return {"prompt": q, "chosen": good, "rejected": bad}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=500)
    p.add_argument("--output_dir", default="data"); a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    pairs = [gen_pair() for _ in range(a.n)]
    split = int(len(pairs) * 0.9)
    for name, data in [("train", pairs[:split]), ("eval", pairs[split:])]:
        with open(out / f"preference_{name}.jsonl", "w") as f:
            for e in data: f.write(json.dumps(e) + "\n")
    print(f"\u2705 Generated {len(pairs)} preference pairs")
    print(f"   Train: {split}, Eval: {len(pairs)-split}")
    print(f"   Avg chosen length: {np.mean([len(p['chosen'].split()) for p in pairs]):.0f} words")
    print(f"   Avg rejected length: {np.mean([len(p['rejected'].split()) for p in pairs]):.0f} words")

if __name__ == "__main__": main()
