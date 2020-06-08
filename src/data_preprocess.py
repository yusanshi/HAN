import pandas as pd
from tqdm import tqdm
from os import path
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import csv
from pathlib import Path
from config import Config


def parse_news(source, target, category2int_path, subcategory2int_path, word2int_path, mode):
    """
    Parse news for training set and test set
    Args:
        source: source news file
        target: target news file
        if mode == 'train':
            category2int_path, subcategory2int_path, word2int_path: Path to save
        elif mode == 'test':
            category2int_path, subcategory2int_path, word2int_path: Path to load from
    """
    print(f"Parse {source}")
    news = pd.read_table(source,
                         header=None,
                         usecols=range(5),
                         names=['id', 'category', 'subcategory', 'title', 'abstract'
                                ])
    # TODO
    news.fillna(' ', inplace=True)
    parsed_news = pd.DataFrame(columns=[
        'id', 'category', 'subcategory', 'title', 'abstract'
    ])

    if mode == 'train':
        category2int = {}
        subcategory2int = {}
        word2int = {}
        word2freq = {}

        for row in news.itertuples(index=False):
            if row.category not in category2int:
                category2int[row.category] = len(category2int)
            if row.subcategory not in subcategory2int:
                subcategory2int[row.subcategory] = len(subcategory2int)

            for w in word_tokenize(row.title.lower()):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1
            for sent in sent_tokenize(row.abstract.lower()):
                for w in word_tokenize(sent):
                    if w not in word2freq:
                        word2freq[w] = 1
                    else:
                        word2freq[w] += 1

        for k, v in word2freq.items():
            if v >= Config.word_freq_threshold:
                word2int[k] = len(word2int) + 1
    else:
        category2int = dict(pd.read_table(category2int_path).values.tolist())
        subcategory2int = dict(pd.read_table(
            subcategory2int_path).values.tolist())
        # na_filter=False is needed since nan is also a valid word
        word2int = dict(
            pd.read_table(word2int_path, na_filter=False).values.tolist())

    word_total = 0
    word_missed = 0
    with tqdm(total=len(news),
              desc="Parsing categories and words") as pbar:
        for row in news.itertuples(index=False):
            try:
                new_row = [
                    row.id,
                    category2int[row.category],
                    subcategory2int[row.subcategory],
                    [],
                    []
                ]
            except KeyError:
                print('Warning: (sub)category not found, dropped!')
                continue

            for w in word_tokenize(row.title.lower()):
                word_total += 1
                if w in word2int:
                    new_row[3].append(word2int[w])
                else:
                    new_row[3].append(0)
                    word_missed += 1

            for sent in sent_tokenize(row.abstract.lower()):
                new_row[4].append([])
                for w in word_tokenize(sent):
                    word_total += 1
                    if w in word2int:
                        new_row[4][-1].append(word2int[w])
                    else:
                        new_row[4][-1].append(0)
                        word_missed += 1

            parsed_news.loc[len(parsed_news)] = new_row

            pbar.update(1)

    print(
        f'Out-of-Vocabulary rate in {mode} set: {word_missed/word_total:.4f}')
    parsed_news.to_csv(target, sep='\t', index=False)

    if mode == 'train':
        pd.DataFrame(category2int.items(),
                     columns=['category', 'int']).to_csv(category2int_path,
                                                         sep='\t',
                                                         index=False)
        print(
            f'Please modify `num_categories` in `src/config.py` into {len(category2int)}'
        )

        pd.DataFrame(subcategory2int.items(),
                     columns=['subcategory', 'int']).to_csv(subcategory2int_path,
                                                            sep='\t',
                                                            index=False)
        print(
            f'Please modify `num_subcategories` in `src/config.py` into {len(subcategory2int)}'
        )

        pd.DataFrame(word2int.items(), columns=['word',
                                                'int']).to_csv(word2int_path,
                                                               sep='\t',
                                                               index=False)
        print(
            f'Please modify `num_words` in `src/config.py` into 1 + {len(word2int)}'
        )


def generate_word_embedding(source, target, word2int_path):
    """
    Generate from pretrained word embedding file
    If a word not in embedding file, initial its embedding by N(0, 1)
    Args:
        source: path of pretrained word embedding file, e.g. glove.840B.300d.txt
        target: path for saving word embedding. Will be saved in numpy format
        word2int_path: vocabulary file when words in it will be searched in pretrained embedding file
    """
    # na_filter=False is needed since nan is also a valid word
    word2int = dict(
        pd.read_table(word2int_path, na_filter=False).values.tolist())
    source_embedding = pd.read_table(source,
                                     index_col=0,
                                     sep=' ',
                                     header=None,
                                     quoting=csv.QUOTE_NONE)
    target_embedding = np.random.normal(size=(1 + len(word2int),
                                              Config.word_embedding_dim))
    target_embedding[0] = 0
    word_missed = 0
    with tqdm(total=len(word2int),
              desc="Generating word embedding from pretrained embedding file"
              ) as pbar:
        for k, v in word2int.items():
            if k in source_embedding.index:
                target_embedding[v] = source_embedding.loc[k].tolist()
            else:
                word_missed += 1

            pbar.update(1)

    print(
        f'Rate of word missed in pretrained embedding: {word_missed/len(word2int):.4f}'
    )
    np.save(target, target_embedding)


if __name__ == '__main__':
    train_dir = './data/train'
    test_dir = './data/test'

    print('Process data for training')

    print('Parse news')
    parse_news(path.join(train_dir, 'news_split.tsv'),
               path.join(train_dir, 'news_parsed.tsv'),
               path.join(train_dir, 'category2int.tsv'),
               path.join(train_dir, 'subcategory2int.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               mode='train')

    print('Generate word embedding')
    generate_word_embedding(
        f'./data/glove/glove.840B.{Config.word_embedding_dim}d.txt',
        path.join(train_dir, 'pretrained_word_embedding.npy'),
        path.join(train_dir, 'word2int.tsv'))

    print('\nProcess data for evaluation')

    print('Parse news')
    parse_news(path.join(test_dir, 'news_split.tsv'),
               path.join(test_dir, 'news_parsed.tsv'),
               path.join(train_dir, 'category2int.tsv'),
               path.join(train_dir, 'subcategory2int.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               mode='test')
