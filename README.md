# HAN

An implementation of [HAN](https://www.aclweb.org/anthology/N16-1174.pdf) (*Hierarchical Attention Networks for Document Classification*) in PyTorch.

## Get started

Basic setup.

```bash
git clone https://github.com/yusanshi/HAN
cd HAN
pip3 install -r requirements.txt
```

Download and preprocess the data.

```bash
mkdir data && cd data
# Download GloVe pre-trained word embedding
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
sudo apt install unzip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip

# Download MIND-small dataset
# By downloading the dataset, you agree to the [Microsoft Research License Terms](https://go.microsoft.com/fwlink/?LinkID=206977). For more detail about the dataset, see https://msnews.github.io/.
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_val.zip
mkdir MIND
unzip MINDsmall_train.zip -d MIND/train
unzip MINDsmall_dev.zip -d MIND/val
rm MINDsmall_*.zip

sort -u MIND/train/news.tsv MIND/val/news.tsv -o news_merged.tsv

# Split it
shuf news_merged.tsv -o news_shuffled.tsv
split -l $[ $(wc -l news_shuffled.tsv | cut -d" " -f1) * 80 / 100 ] news_shuffled.tsv
mkdir train
mv xaa train/news_split.tsv
mkdir test
mv xab test/news_split.tsv

# Preprocess data into appropriate format
cd ..
python3 src/data_preprocess.py
# Remember you shoud modify `num_*` in `src/config.py` by the output of `src/data_preprocess.py`
```

Run.

```bash
# Train and save checkpoint into `checkpoint/` directory
python3 src/train.py
# Load latest checkpoint and evaluate on the test set
python3 src/evaluate.py

# or

chmod +x run.sh
./run.sh
```

You can visualize metrics with TensorBoard.

```bash
tensorboard --logdir=runs
```

> Tip: by adding `REMARK` environment variable, you can make the runs name in TensorBoard more meaningful. For example, `REMARK=learning_rate-0.001 python3 src/train.py`.

## Results


**TODO**


## Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
