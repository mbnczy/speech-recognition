import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

def human_readable_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def desc_wc_stat(df, text_col_name, save_as, title):
    df['text_wc'] = df[text_col_name].str.split().str.len()
    desc = df['text_wc'].describe()
    summary_ticks = [desc['min'], desc['25%'], desc['50%'], desc['75%'], desc['max']]
    summary_ticks = np.round(summary_ticks, 0)
    
    hist_data = df['text_wc']
    default_ticks = np.linspace(1, hist_data.max(), num=10)
    combined_ticks = sorted(set(np.round(np.concatenate([default_ticks, summary_ticks]), 0)))
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1])
    
    sns.histplot(hist_data, bins=30, kde=True, color='skyblue', stat='density', ax=axes[0])
    axes[0].set_xlim(0, desc['max'])
    axes[0].set_xlabel("", fontsize=12)
    axes[0].set_title(title, fontsize=14)
    axes[0].set_ylabel("density", fontsize=12)
    axes[0].grid(True, axis='x', linestyle='--', alpha=0.9)
    axes[0].set_xticks(combined_ticks)
    
    sns.boxplot(x=hist_data, color='lightblue', ax=axes[1])
    axes[1].set_xlim(0, desc['max'])
    axes[1].set_xlabel("word count", fontsize=12)
    axes[1].set_yticks([])
    axes[1].grid(True, axis='x', linestyle='--', alpha=0.9)
    axes[1].set_xticks(combined_ticks)
    
    for ax in axes:
        for label in ax.get_xticklabels():
            tick_value = float(label.get_text())
            if tick_value in summary_ticks:
                label.set_color("coral")

    stats_text = f"- Min: {desc['min']:.1f}\n- 25%: {desc['25%']:.1f}\n- Median: {desc['50%']:.1f}\n- 75%: {desc['75%']:.1f}\n- Max: {desc['max']:.1f}\n- Mean: {desc['mean']:.1f}\n- Std: {desc['std']:.1f}"
    axes[0].legend([stats_text], loc='upper right', fontsize=10, frameon=True, title="Summary Stats")

    
    plt.tight_layout()
    plt.savefig(save_as, dpi=600)
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default=os.environ.get("DATA_FOLDER", "../large_data/Up First"))
    args = parser.parse_args()
    
    save_as = f"{os.path.basename(data_folder)}-{datetime.now().strftime('%Y_%m_%d')}"

    dfs = []
    
    for root, dirs, files in tqdm(
        os.walk(args.data_folder),
        desc="Directories"
    ):
        if len(dirs) == 0 and ".ipynb_checkpoints" not in root and "__MACOSX" not in root:
            for file in tqdm(
                files,
                desc=f"Files in {os.path.basename(root)}",
                leave=False
            ):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(root,file))
                    df["id"]=df.sentence_id.astype(str).apply(
                        lambda s_id: f"{os.path.basename(root).replace(' ','_')}--{os.path.splitext(file)[0].replace(' ','_')}--{s_id}--{datetime.now().strftime('%Y_%m_%d')}"
                    )
                    dfs.append(df[["id","sentence"]])
                    
                    
    df = pd.concat(dfs, axis=0)
    df["group_key"] = df["id"].apply(lambda uid: uid.split('--')[1].split('_')[0])
    
    merged_df = (
        df.groupby("group_key", as_index=False)
          .agg(full_show_text=('sentence', ' '.join))
    )
    
    df = df.merge(merged_df, on="group_key", how="left").drop(columns=["group_key"])
    df = df[["id","full_show_text","sentence"]]
    
    df.to_parquet(save_as+".parquet", index=False)
    df.to_csv(save_as+".csv", index=False)
    
    print(f"{len(df)} rows")
    print(f"{save_as}.parquet file size: {human_readable_size(os.path.getsize(save_as + '.parquet'))}")
    print(f"{save_as}.csv file size: {human_readable_size(os.path.getsize(save_as + '.csv'))}")
    
    desc_wc_stat(
        df = df,
        text_col_name = "sentence",
        save_as = save_as+"-text_desc_stat.png",
        title = f"{os.path.basename(args.data_folder)} - Distribution of Sentence Lengths"
    )
    desc_wc_stat(
        df = df.drop_duplicates(subset="full_show_text"),
        text_col_name = "full_show_text",
        save_as = save_as+"-sent_desc_stat.png",
        title = f"{os.path.basename(args.data_folder)} - Distribution of Full Show Text Lengths"
    )
