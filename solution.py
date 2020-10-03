import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
import gensim
from tqdm import tqdm
import random

DATA_PATH = "./Data Scientist Evaluation Test.xlsx"


class JobProblem:
    def __init__(self,
                 filepath,
                 stopwords=STOP_WORDS,
                 vocab_size=300,
                 min_count=4,
                 max_len=20,
                 epochs=70,
                 window_size=15,
                 num_workers=4
                 ):
        self.data = self.load_data(filepath)
        self.titles = self.data.title.unique()
        self.stopwords = stopwords
        self.vocab_size = vocab_size
        self.min_count = min_count
        self.max_len = max_len
        self.window_size = window_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.train_data = None
        self.train_dict = None
        self.test_data = None
        self.test_dict = None
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vocab_size,
                                                   min_count=min_count,
                                                   epochs=epochs,
                                                   window=window_size,
                                                   num_workers=num_workers)

    def load_data(self, file):
        frame = pd.read_excel(file, index_col=0)
        frame.columns = ['req', 'desc', 'sen', 'title']
        return frame

    def sample_titles(self, num=10, replace=False):
        titles = list(self.data['title'].value_counts()[:num].index)
        if replace:
            self.data = self.data[self.data['title'].isin(titles)]
            self.data = self.data.reset_index(drop=True)
        # Memory DangOFang!
        else:
            self.sample = self.data[self.data['title'].isin(titles)]
            self.sample = self.sample.reset_index(drop=True)
        self.titles = titles

    def split(self, train_size=0.8):
        len_test = int(len(self.data) * (1 - train_size))
        self.test_ids = list(np.random.choice(range(len(self.data)), size=len_test, replace=False))
        self.train_ids = [i for i in range(len(self.data)) if i not in self.test_ids]
        self.train_data = self.data.iloc[self.train_ids]
        self.test_data = self.data.iloc[self.test_ids]
        delattr(self, 'data')
        print("There is Train/Test data now.")

    @staticmethod
    def tokenize(text, stopwords, max_len=20):
        processed = gensim.utils.simple_preprocess(text, max_len=max_len)  # O(len(text))
        return [token for token in processed if token not in stopwords]  # O(tokens + 1) >> O(n)

    def make_dict(self, column_name='desc'):
        if (self.train_data is not None) and (self.test_data is not None):
            self.train_dict = {}
            self.test_dict = {}
            for t in tqdm(self.titles, desc='Making Dict'):
                self.train_dict[t] = self.train_data[column_name].to_list()
                self.test_dict[t] = self.test_data[column_name].to_list()
        else:
            self.split()
            self.make_dict()

    def tagg_dict(self):
        if (self.train_dict is not None) and (self.test_dict is not None):
            tagged_train = {}
            tagged_test = {}
            tag = 0
            for tit, corp in self.train_dict.items():
                tagged_train[tit] = [
                    gensim.models.doc2vec.TaggedDocument(self.tokenize(text, self.stopwords, self.max_len), [i + tag])
                    for i, text in enumerate(corp)
                ]
                tag += len(corp)

            for tit, corp in self.test_dict.items():
                tagged_test[tit] = [self.tokenize(text, self.stopwords, self.max_len) for i, text in enumerate(corp)]
                tag += len(corp)

            self.train_tagg = tagged_train
            self.test_tagg = tagged_test
        else:
            self.make_dict()
            self.tagg_dict()
        self.train_corpus = [doc for doclist in list(self.train_tagg.values()) for doc in doclist]
        print("The Train Corpus is now Ready.")

    def preprocess(self):
        pass

    def train(self):
        if self.train_corpus:
            self.model.build_vocab(self.train_corpus)
            self.model.train(
                self.train_corpus,
                total_examples=self.model.corpus_count,
                epochs=self.model.epochs
            )

    def evaluate(self):
        self.test_vectors = {}
        self.summary = {}
        for job, docs in self.test_tagg.items():  # docs = [[doc1],[doc2],...] ListofList
            self.test_vectors[job] = [self.model.infer_vector(doc) for doc in list(docs)]  # doc >> list
            accuracy = (len(self.test_data) / len(self.test_vectors[job])) * 100
            print("In job {} : {}% of documents are Compatible. ".format(job, accuracy))
            self.summary[job] = len(self.test_vectors[job])
        labels = []
        vectors = []
        for job in self.train_dict.keys():
            for tag in [job] * self.summary[job]:
                labels.append([tag])
            for v in self.test_vectors[job]:
                vectors.append(list(v))
        self.raw_test_vecs = vectors
        self.raw_test_labels = labels

    def save_results_for_visualization(self):
        pass

    def calculate_similarities(self):  # O(n ^ n) >> Time Consuming (PairWise Distance >> Cosine Similarity)
        train = {}
        for job, doc in self.train_dict.items():
            train[job] = [self.tokenize(text, self.stopwords, self.max_len) for i, text in enumerate(doc)]

        job_id = {id: job for id, job in enumerate(self.titles)}
        self.job_id = job_id
        job_pairs = {tuple(sorted([id_, id__])): [] for id_ in job_id for id__ in job_id}

        for pair in job_pairs:
            # Internal Job Similarities
            if pair[0] == pair[1]:
                job_pairs[pair] = [(doc, train[job_id[pair[0]]][i])
                                   for index, doc in enumerate(list(self.test_tagg[job_id[pair[0]]]))
                                   for i in range(index + 1, len(list(self.test_tagg[job_id[pair[0]]])))
                                   ]
            # Cross Job Similiraties
            else:
                job_pairs[pair] = [(doc_, doc__)
                                   for doc_ in list(self.test_tagg[job_id[pair[0]]])
                                   for doc__ in list(self.test_tagg[job_id[pair[1]]])
                                   ]

        similarities = {pi: [] for pi in job_pairs}
        for id_ in tqdm(job_id, desc="Calculating Similarities"):
            for id__ in job_id:
                similarities[tuple(sorted([id_, id__]))] = [

                    self.model.docvecs.similarity_unseen_docs(self.model, pair[0], pair[1])
                    for pair in random.sample(
                        job_pairs[tuple(sorted([id_, id__]))],
                        len(job_pairs[tuple(sorted([id_, id__]))]))[:500]
                ]
        self.similarities = similarities
        print("The Similarities Calculated Successfully. Check obj.similarities")

    def report_results(self, verbose=True):
        report = {}
        for k, v in self.similarities.items():
            report[(self.job_id[k[0]], self.job_id[k[1]])] = sum(v) / len(v)
        if verbose:
            print(report)
        frame = pd.DataFrame(list(report.keys()))
        frame.columns = ['job1', 'job2']
        frame['similarities'] = report.values()
        frame = frame.style.background_gradient(cmap='turbo')
        self.result_frame = frame
        return frame

    def get_taggs(self, job, n=20):
        job_ = job.lower()
        job_ = job_.split()
        tag_recommends = self.model.wv.most_similar(positive=job_, topn=n)
        for i in tag_recommends:
            print(i)

    def __len__(self):
        return self.data.__len__()

if __name__ == '__main__':
    solver = JobProblem(DATA_PATH)
    solver.sample_titles(num=10, replace=True)
    print(solver.titles)
    solver.tagg_dict()
    solver.train()
    solver.evaluate()
    solver.calculate_similarities()
    solver.report_results(verbose=True)
    job = 'Product Manager'
    solver.get_taggs(job)

