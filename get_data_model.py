import pandas as pd
import numpy as np
import pyBigWig
import bioframe as bf
import pyfaidx
import logging
from tangermeme.utils import one_hot_encode
import torch
from scipy.stats import pearsonr
from grelu.io.genome import get_genome
from torchmetrics.functional import pearson_corrcoef
from torch.utils.data import Dataset
import pytorch_lightning as pl


from itertools import groupby

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_n_intervals(genome, chroms, min_n_count=10):
    n_intervals = []

    for chrom in chroms:
        seq = str(genome[chrom])
        starts = np.where(np.array(list(seq)) == 'N')[0]

        if len(starts) == 0:
            continue

        diffs = np.diff(starts)
        split_points = np.where(diffs > 1)[0] + 1
        intervals = np.split(starts, split_points)

        intervals = [(i[0], i[-1] + 1) for i in intervals if len(i) >= min_n_count]
        n_intervals.extend([(chrom, start, end) for start, end in intervals])

    return pd.DataFrame(n_intervals, columns=["chrom", "start", "end"]) if n_intervals else None


def get_defined_intervals(genome,chroms, min_length=None, min_n_count=10):

    intervals = pd.DataFrame([(chrom, 0, len(genome[chrom])) for chrom in chroms],
                             columns=["chrom", "start", "end"])

    n_intervals = get_n_intervals(genome, chroms, min_n_count)
    if n_intervals is not None:
        intervals = bf.subtract(intervals, n_intervals)

    if min_length:
        intervals = intervals[(intervals.end - intervals.start) > min_length]

    return intervals


def create_interval_n(interval_df, seq_length, bin_size,target_length, n):
    chrom_intervals = {}
    for idx, row in interval_df.iterrows():
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        intervals = []
        for inter_start in range(start, end - seq_length,target_length*bin_size):
            inter_end = min(inter_start + seq_length ,end)
            predic_start = inter_start + bin_size*n
            predic_end = inter_end - n*bin_size  
            intervals.append((chrom, inter_start, inter_end, predic_start, predic_end))
        new_intervals_df = pd.DataFrame(intervals, columns=["chrom", "start", "end", "predic_start", "predic_end"])
        if chrom in chrom_intervals:
            chrom_intervals[chrom] =pd.concat([chrom_intervals[chrom], new_intervals_df], ignore_index=True)
        else:
            chrom_intervals[chrom] = new_intervals_df
    return chrom_intervals



def expression_bw(expression_values, bin_size):
    expression_values = np.nan_to_num(expression_values)
    valid_size = (expression_values.size // bin_size) * bin_size
    if valid_size > 0:
        expression_values = expression_values[:valid_size]  
        binned_expression = expression_values.reshape(-1, bin_size).mean(axis=1)
    else:
        binned_expression = np.array([expression_values.mean()]) 
    return binned_expression.astype(np.float32)



def expression_val_histone(df,bin_size, bws):
    
    results = {}  

    for idx, bw in enumerate(bws):
        bw = pyBigWig.open(bw) 
        chrom_sizes = bw.chroms()  
        binned_expression_list = []
        max_bins = 0

        for _, row in df.iterrows():
            chrom, start, end = row["chrom"], row["predic_start"], row["predic_end"]
            chrom_length = chrom_sizes.get(chrom, 0)

            if chrom_length == 0 or start >= chrom_length or end <= 0:
                binned_expression_list.append(np.zeros(1, dtype=np.float32))
                continue

            start, end = max(0, start), min(chrom_length, end)

            expression_values = bw.values(chrom, start, end, numpy=True)
            expression_values = np.nan_to_num(expression_values)  

            valid_size = (expression_values.size // bin_size) * bin_size
            if valid_size > 0:
                expression_values = expression_values[:valid_size]
                binned_expression = expression_values.reshape(-1, bin_size).mean(axis=1)
            else:
                binned_expression = np.array([expression_values.mean()], dtype=np.float32)

            max_bins = max(max_bins, len(binned_expression))
            binned_expression_list.append(binned_expression)

        for i in range(len(binned_expression_list)):
            diff = max_bins - len(binned_expression_list[i])
            if diff > 0:
                binned_expression_list[i] = np.pad(binned_expression_list[i], (0, diff), mode='constant', constant_values=0)

        results[f"bw_{idx}"] = binned_expression_list  

        bw.close()  

    return pd.DataFrame(results, index=df.index) 


def get_mapability(df,bin_size, bws):
    results = {}  

    for idx, bw in enumerate(bws):  
        bw = pyBigWig.open(bw) 
        chrom_sizes = bw.chroms()
        mapability_list = []
        max_bins = 0

        for _, row in df.iterrows():
            chrom, start, end = row["chrom"], row["predic_start"], row["predic_end"]
            chrom_length = chrom_sizes.get(chrom, 0)

            if chrom_length == 0 or start >= chrom_length or end <= 0:
                mapability_list.append(np.zeros(1, dtype=bool))
                continue

            start, end = max(0, start), min(chrom_length, end)

            mapability_values = bw.values(chrom, start, end, numpy=True)
            mapability_values = np.nan_to_num(mapability_values) 

            valid_size = (mapability_values.size // bin_size) * bin_size
            if valid_size > 0:
                mapability_values = mapability_values[:valid_size]
                mapability = mapability_values.reshape(-1, bin_size).mean(axis=1)
            else:
                mapability = np.array([mapability_values.mean()])

            mapability = (mapability > 0.9).astype(float)

            max_bins = max(max_bins, len(mapability))
            mapability_list.append(mapability)

        for i in range(len(mapability_list)):
            diff = max_bins - len(mapability_list[i])
            if diff > 0:
                mapability_list[i] = np.pad(mapability_list[i], (0, diff), mode='constant', constant_values=0)

        results[f"mappability_{idx}"] = mapability_list  

        bw.close()  

    return pd.DataFrame(results, index=df.index)  



def process_sequences(ds, genome):
    
    return ds.map(
        lambda x: {
            "input_oh": one_hot_encode(
                genome.get_seq(x["chrom"], int(x["start"]) + 1, int(x["end"])).seq.upper(),
            ).to(torch.int8),
            "mask": 1 if any(c.islower() for c in genome.get_seq(x["chrom"], int(x["start"]) + 1, int(x["end"])).seq) else 0
        }
    )



def get_value_bw( signal_bw_paths,mappability_bw_path,df_train,df_val, df_test, stage, bin_size):
    
    if stage == "fit":
        for i, bw_path in enumerate(signal_bw_paths):
            train_expression = expression_val_histone(df_train, bin_size=bin_size, bws=[bw_path])
            df_train.loc[:, f'labels_{i}'] = train_expression

            val_expression = expression_val_histone(df_val, bin_size=bin_size, bws=[bw_path])
            df_val.loc[:, f'labels_{i}'] = val_expression

        train_map = get_mapability(df_train, bin_size=bin_size, bws=mappability_bw_path)
        df_train.loc[:, 'map'] = train_map

        val_map = get_mapability(df_val, bin_size=bin_size, bws=mappability_bw_path)
        df_val.loc[:, 'map'] = val_map

    elif stage == "test":
        for i, bw_path in enumerate(signal_bw_paths):
            test_expression = expression_val_histone(df_test, bin_size=bin_size, bws=[bw_path])
            df_test.loc[:, f'labels_{i}'] = test_expression

        test_map = get_mapability(df_test, bin_size=bin_size, bws=mappability_bw_path)
        df_test.loc[:, 'map'] = test_map

    return df_train, df_val, df_test



def load_contigs(partition_file, split='train'):
    df = pd.read_csv(partition_file, sep='\t', header=None, names=['contig', 'set'])
    selected = df[df['set'] == split]
    return selected['contig'].tolist()



class BorzoiDataset(Dataset):
    def __init__(self, contigs, data_dir):
        self.files = [f"{data_dir}/{contig}.npz" for contig in contigs]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        seq = torch.tensor(data["sequence"]).float()
        target = torch.tensor(data["target"]).float()
        mask = torch.tensor(data["mask"]).float()
        return {"input_oh": seq, "labels": target, "mask": mask}
    

class DataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        if stage == "fit":
            train_contigs = load_contigs("path/to/contig_components.txt", "train")
            val_contigs = load_contigs("path/to/contig_components.txt", "valid")
            self.train_ds = BorzoiDataset(train_contigs, "path/to/hg38")
            self.val_ds = BorzoiDataset(val_contigs, "path/to/hg38")

        if stage == "test":
            test_contigs = load_contigs("path/to/contig_components.txt", "test")
            self.test_ds = BorzoiDataset(test_contigs, "path/to/hg38")



