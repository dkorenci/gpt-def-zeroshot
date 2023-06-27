'''
Utils for multi-category classification (N-way classification per category)
'''
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, precision_score, accuracy_score
from functools import partial

from experiments.corpus_utils import Corpus

f1_macro = partial(f1_score, average='macro')
recall_macro = partial(recall_score, average='macro')
prec_macro = partial(precision_score, average='macro')

def check_categories(categories: List[str]):
    if len(set(categories)) != len(categories):
        raise ValueError('categories must be unique')
    if len(categories) == 0:
        raise ValueError('categories must be non-empty')
    for c in categories:
        if not isinstance(c, str) or len(c) == 0:
            raise ValueError('each category must be a non-empty string')

class MultiCategoryScorer:
    '''
    Helper class for storing and scoring multi-category classification results.
    '''

    def __init__(self, gold_corpus: Corpus, categories: List[str], verbose=False):
        self.gold_corpus = gold_corpus
        self.categories = categories
        check_categories(categories)
        # dataframe for storing scores
        self._df_scores = pd.DataFrame(columns=categories + ['text_id', 'text'])
        self._verbose = verbose

    def add(self, text_id: object, scores: Dict[str, int]):
        if set(scores.keys()) != set(self.categories):
            raise ValueError('scores keys must match categories exactly')
        if text_id not in self.gold_corpus:
            raise ValueError(f'text_id {text_id} not in "gold" corpus')
        new_row = pd.DataFrame([{**scores, 'text_id': text_id}])
        self._df_scores = pd.concat([self._df_scores, new_row], ignore_index=True)

    def coprediction_matrix(self):
        df = self._predictions_as_dframe()
        return pd.DataFrame({categ1: [sum(df[categ1] & df[categ2]) for categ2 in self.categories] for categ1 in self.categories},
                            index=self.categories)

    def score(self, per_category=True, binary=True):
        '''
        :param per_category: if True, print scores for each category, not only the average
        :param binary: if True, use binary metrics (F1, P, R), otherwise macro-averaged (F1_macro, P_macro, R_macro)
        :return:
        '''
        if not binary:
            score_fns = {'F1_macro': f1_macro, 'P_macro': prec_macro, 'R_macro': recall_macro,
                         'ACC': accuracy_score, '_MCC': matthews_corrcoef}
        else:
            score_fns = {'F1': f1_score, 'P': precision_score, 'R': recall_score,
                         'ACC': accuracy_score, '_MCC': matthews_corrcoef}
        pred_df = self._predictions_as_dframe()
        ref_df = self._gold_labels_as_dframe()
        scores = []
        cat_scores = {fname: [] for fname in score_fns}
        for i, cat in enumerate(self.categories):
            pred = list(pred_df[cat])
            true = list(ref_df[cat])
            for fname in cat_scores:
                cat_scores[fname].append(score_fns[fname](true, pred))
            scores.append(cat_scores)
            if per_category and self._verbose:
                print(f'{cat:30}: {";".join([f"{fname}:{cat_scores[fname][-1]:.3f}" for fname in sorted(list(score_fns.keys()))])}')
        scores_avg = {fname: np.average([scr[fname] for scr in scores]) for fname in score_fns}
        scrprnt = ('' if not per_category else f'{"AVERAGE":30}: ') + \
                  ';'.join([f'{fname}:{scores_avg[fname]:.3f}' for fname in sorted(list(score_fns.keys()))])
        if self._verbose: print(scrprnt)
        return scores_avg, cat_scores

    def _predictions_as_dframe(self):
        ''' DataFrame with current predictions. Must have a 'text_id' column. '''
        return self._df_scores.copy()

    def _gold_labels_as_dframe(self):
        txts = self.gold_corpus.get_texts(self._df_scores['text_id'])
        data = [
            { **{categ: getattr(txto, categ) for categ in self.categories}, **{'text_id': txto.id, 'text': txto.text} }
            for txto in txts
        ]
        return pd.DataFrame(data)
