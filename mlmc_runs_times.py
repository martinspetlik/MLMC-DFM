import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing your 0000_STDOUT, 0001_STDOUT, ... files
LOG_DIR = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed_cost/jobs"

# Regex patterns for each timing line
patterns = {
    "dfn_creation_time": r"DFN creation time\s+([0-9.]+)",
    "fine_bulk_creation_time": r"FINE BULK creation time\s+([0-9.]+)",
    "fine_sample_flow_time": r"FINE SAMPLE FLOW run time\s+([0-9.]+)",
    "coarse_dfn_prep_time": r"COARSE DFN prep time\s+([0-9.]+)",
    "homogenization_time": r"homogenization time\s+([0-9.]+)",
    "coarse_sample_flow_time": r"COARSE SAMPLE FLOW run time\s+([0-9.]+)"
}

cols = [
    "dfn_creation_time",
    "fine_bulk_creation_time",
    "fine_sample_flow_time",
    "coarse_dfn_prep_time",
    "homogenization_time",
    "coarse_sample_flow_time"
]

given_total_time_by_level = pd.Series({
    1: 221,
    2: 2387
})


# Identifier regex: example L01_S0000099
id_pattern = re.compile(r"L(\d{2})_S(\d{7,8})")

def parse_stdout_file(path):
    """Parse a single STDOUT file for timings and sample metadata."""
    results = {key: None for key in patterns.keys()}
    levels = []
    samples = []

    with open(path, "r") as f:
        content = f.read()

        # Extract timings
        for key, pat in patterns.items():
            match = re.search(pat, content)
            if match:
                results[key] = float(match.group(1))

        # print("content ", content[:500])
        # print("L02_S0000050" in content)
        #
        #
        # id_pattern = re.compile(r"L(\d{2})_S(\d{7,8})")
        #
        # matches = id_pattern.findall(content)
        # print("Matches:", matches)

        # Extract all Lxx_Syyyyyyyy occurrences
        for m in id_pattern.finditer(content):
            level = int(m.group(1))
            sample = m.group(2)
            levels.append(level)
            samples.append(sample)

    # Store unique or first-occurrence metadata
    if levels:
        results["levels_found"] = sorted(set(levels))
        results["primary_level"] = levels[0]  # You can change logic if needed
    else:
        results["levels_found"] = []
        results["primary_level"] = None

    if samples:
        results["samples_found"] = sorted(set(samples))
    else:
        results["samples_found"] = []

    return results


# Iterate over all *_STDOUT files
records = []
for filename in sorted(os.listdir(LOG_DIR)):
    if filename.endswith("_STDOUT"):
        file_index = int(filename.split("_")[0])  # e.g., "0003" → 3
        file_path = os.path.join(LOG_DIR, filename)
        print("file name ", filename)

        data = parse_stdout_file(file_path)
        print("data ", data)
        data["file"] = filename
        data["index"] = file_index
        records.append(data)

df = pd.DataFrame(records).sort_values("index")
#
# # Select level
# level = 2
# df = df[df["primary_level"] == level].copy()
# ######

print("dfn_creation_time ", df["dfn_creation_time"])
print("fine_bulk_creation_time ", df["fine_bulk_creation_time"])

# df_level_2 = df[df["primary_level"] == 2]
# print("df_level_2 " ,df_level_2)

mean_by_level = df.groupby("primary_level").mean(numeric_only=True)
mean_by_level["sum_of_means"] = mean_by_level.sum(axis=1)
total_mean_cost_by_level = mean_by_level["sum_of_means"]
print(total_mean_cost_by_level)

ratios = mean_by_level[cols].div(mean_by_level[cols].sum(axis=1), axis=0)

print('fractions ', ratios[cols].sum(axis=1))

scaled_values = ratios.mul(given_total_time_by_level, axis=0)



plt.figure(figsize=(12, 6))

# level = 1
# row = scaled_values.loc[level]
#
# plt.figure(figsize=(10,5))
# plt.bar(cols, row)
# plt.ylabel("Time [s]")
# plt.title(f"Component breakdown for Level {level} (total = {given_total_time_by_level[level]} s)")
# plt.xticks(rotation=45)
# plt.show()

levels = scaled_values.index
bottom = None

plt.figure(figsize=(12, 6))

for col in cols:
    if bottom is None:
        plt.bar(levels, scaled_values[col], label=col)
        bottom = scaled_values[col].copy()
    else:
        plt.bar(levels, scaled_values[col], bottom=bottom, label=col)
        bottom = bottom + scaled_values[col]

plt.xlabel("Level")
plt.ylabel("Fraction of total mean runtime")
plt.title("Fractional contribution of runtime components per level")
plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
plt.tight_layout()
plt.show()
#

# -------------------------------------------------------
# PLOTTING: NORMALIZED FRACTIONS + TOTAL TIME ANNOTATION
# -------------------------------------------------------

ratios = mean_by_level[cols].div(mean_by_level[cols].sum(axis=1), axis=0)

levels = ratios.index
bottom = None

plt.figure(figsize=(12, 6))

# Plot normalized stacked bars (each sums to 1)
for col in cols:
    if bottom is None:
        plt.bar(levels, ratios[col], label=col)
        bottom = ratios[col].copy()
    else:
        plt.bar(levels, ratios[col], bottom=bottom, label=col)
        bottom = bottom + ratios[col]

# Annotate each bar with the given absolute total time
for level in levels:
    total = given_total_time_by_level[level]
    plt.text(
        x=level,
        y=1.02,                              # slightly above bar
        s=f"Total = {total} s",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

plt.xlabel("Level")
plt.ylabel("Fraction of total time (normalized)")
plt.title("Normalized component contributions with absolute total times")
plt.ylim(0, 1.15)      # leave room for labels
plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
plt.tight_layout()
plt.show()


