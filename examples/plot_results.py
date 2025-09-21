import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("assets/bench.csv")
df = df.sort_values(["N","n_angles","iters"])

plt.figure()
plt.plot(df["mart_ms"], label="GPU MART (ms)")
plt.plot(df["fbp_ms"], label="CPU FBP (ms)")
plt.xlabel("Run #"); plt.ylabel("Milliseconds"); plt.legend(); plt.tight_layout()
plt.savefig("assets/timings.png", dpi=180)

plt.figure()
plt.plot(df["psnr_mart"], label="PSNR MART (dB)")
plt.plot(df["psnr_fbp"], label="PSNR FBP (dB)")
plt.xlabel("Run #"); plt.ylabel("dB"); plt.legend(); plt.tight_layout()
plt.savefig("assets/quality.png", dpi=180)
print("Saved assets/timings.png and assets/quality.png")

