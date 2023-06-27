from pathlib import Path

import pandas as pd
import json
from sklearn.metrics import cohen_kappa_score

from experiments import data_utils
from experiments.data_utils import CONSPIRACY_CAT_IDS
from experiments.classif_utils import MultiCategoryScorer
from settings import EXPERIM_V1_GPT35_RESULTS, EXPERIM_V1_GENERATED_DEFS

ID_COL = 'tweet_id'

# shortucts used in the result dataframes, order aligned with CONSPIRACY_CAT_IDS
EXPERIM_V1_CAT_SHORTCUTS = ['av', 'bmc', 'fv', 'hri', 'ip', 'nwo', 'prc', 'sat', 'sc']

# map shortcuts to cat ids, by index
CAT_SHORTCUT2ID = {EXPERIM_V1_CAT_SHORTCUTS[i]: CONSPIRACY_CAT_IDS[i] for i in range(len(CONSPIRACY_CAT_IDS))}

def load_generated_defs_file(file):
    '''
    Load generated definitions from a file.
    :return: map of category id -> definition
    '''
    # must be in the same order as the definitions in the file
    GENDEF_ORDER_CAT_SHORTCUTS = ['sc', 'bmc', 'av', 'fv', 'ip', 'hri', 'prc', 'nwo', 'sat']
    df = pd.read_csv(file, sep=',')
    res = {}
    for ix in df.index:
        r = df.iloc[ix, 1]
        r = json.loads(r)
        defn = r['choices'][0]['message']['content']
        #print(defn)
        cat = GENDEF_ORDER_CAT_SHORTCUTS[ix]
        res[CAT_SHORTCUT2ID[cat]] = defn
    #print()
    return res

def load_all_gen_defs():
    '''
    :return: map random seed -> generated definitions
    '''
    res = {}
    for seed in range(5):
        file = Path(EXPERIM_V1_GENERATED_DEFS)/f'eg_definitions_{seed}.csv'
        defs = load_generated_defs_file(file)
        res[seed] = defs
    return res

def load_gen_def_result(seed):
    file = Path(EXPERIM_V1_GPT35_RESULTS)/f'eg_results_{seed}.csv'
    return pd.read_csv(file, sep=',')

def gendef_res_coprediction_matrix(df1, df2, chance_correct=True):
    ''' Matrix of category X category co-predictions for the results of zero-shot
     classifiers given as data frames.
     :return: dataframe with rows and columns corresponding to categories,
                and values corresponding co-prediction between two categories zero-shot classifiers.
     '''
    if df1 is not df2:
        # filter out rows that are not in both dataframes
        common_ids = set(df1[ID_COL]).intersection(set(df2[ID_COL]))
        df1 = df1[df1[ID_COL].isin(common_ids)]
        df2 = df2[df2[ID_COL].isin(common_ids)]
    cats = EXPERIM_V1_CAT_SHORTCUTS
    cat2id = CAT_SHORTCUT2ID
    if not chance_correct:
        copred = pd.DataFrame({cat2id[categ1]: [sum(df1[categ1] & df2[categ2]) for categ2 in cats] for categ1 in cats},
                         index=[cat2id[c] for c in cats])
    else:
        copred = pd.DataFrame({cat2id[categ1]: [cohen_kappa_score(df1[categ1], df2[categ2]) for categ2 in cats]
                               for categ1 in cats}, index=[cat2id[c] for c in cats])
    return copred

def gendef_res_coprediction_matrix_4seeds(seed1, seed2):
    df1 = load_gen_def_result(seed1)
    df2 = load_gen_def_result(seed2)
    return gendef_res_coprediction_matrix(df1, df2)

def conspi_res2scorer(res_df):
    '''
    Converts the results dataframe to MultiCategoryScorer object
    :return:
    '''
    test_corpus = data_utils.test_corpus_ment_noment()
    scorer = MultiCategoryScorer(test_corpus, CONSPIRACY_CAT_IDS)
    # iterate rows of the dataframe
    for ix in res_df.index:
        row = res_df.iloc[ix]
        cat_preds = { CAT_SHORTCUT2ID[cat]: row[cat] for cat in EXPERIM_V1_CAT_SHORTCUTS}
        text_id = row[ID_COL]
        scorer.add(text_id, cat_preds)
    return scorer

def load_all_def_scores():
    '''
    :return: map seed -> MultiCategoryScorer
    '''
    res = {}
    for seed in range(5):
        res_df = load_gen_def_result(seed)
        scorer = conspi_res2scorer(res_df)
        res[seed] = scorer
    return res

# adapted category names to match std. ID categories
EXPERIM_GOLDDEFS_ADAPT = ["Suppressed cures: Narratives which propose that effective medications for COVID-19 were available, but whose existence or effectiveness has been denied by authorities, either for financial gain by the vaccine producers or some other harmful intent.",
                         "Behaviour and mind control: Narratives containing the idea that the pandemic is being exploited to control the behavior of individuals, either directly through fear, through laws which are only accepted because of fear, or through techniques which are impossible with today’s technology, such as mind control through microchips.",
                         "Antivax: Narratives that suggest that the COVID-19 vaccines serve some hidden nefarious purpose in this category. Examples include the injection of tracking devices, nanites or an intentional infection with COVID-19, but not concerns about vaccine safety or efficacy, or concerns about the trustworthiness of the producers.",
                         "Fake virus: Narratives saying that there is no COVID-19 pandemic or that the pandemic is just an over-dramatization of the annual flu season. Example intent is to deceive the population in order to hide deaths from other causes, or to control the behavior of the population through irrational fear.",
                         "Intentional pandemic: Narratives claiming that the pandemic is the result of purposeful human action pursuing some illicit goal. Does not include asserting that COVID-19 is a bioweapon or discussing whether it was created in a labora-tory since this does not prelude the possibility that it was released accidentally.",
                         "Harmful radiation influence: Narratives that connect COVID-19 to wireless transmissions, especially from 5 G equipment, claiming for example that 5 G is deadly and that COVID-19 is a coverup, or that 5 G allows mind control via microchips injected in the bloodstream.",
                         "Population reduction control: Conspiracy theories on population reduction or population growth control suggest that either COVID-19 or the vaccines are being used to reduce population size, either by killing people or by rendering them infertile. In some cases, this is directed against specific ethnic groups.",
                         "New world order: New World Order (NWO) is a preexisting conspiracy theory which deals with the secret emerging totalitarian world government. In the context of the pandemic, this usually means that COVID-19 is being used to bring about this world government through fear of the virus or by taking away civil liberties, or some other, implausible ideas such as mind control.",
                         "Satanism: Narratives in which the perpetrators are alleged to be some kind of satanists, perform objectionable rituals, or make use of occult ideas or symbols. May involve harm or sexual abuse of children, such as the idea that global elites harvest adrenochrome from children."
                          ]

def golddefs_as_catmap():
    cat2def = {}
    # split and map categ to def
    for deftxt in EXPERIM_GOLDDEFS_ADAPT:
        cat, defn = deftxt.split(':')
        cat = cat.strip().lower().replace(' ', '_')
        defn = defn.strip()
        cat2def[cat] = defn
    # map defs to std. cat ids
    res = {}
    for cat in CONSPIRACY_CAT_IDS:
        catn = cat.lower()
        assert catn in cat2def
        res[cat] = cat2def[catn]
    return res

if __name__ == '__main__':
    golddefs_as_catmap()