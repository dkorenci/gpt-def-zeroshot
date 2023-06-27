import csv
from pathlib import Path
import pandas as pd
from pandas import DataFrame

from experiments.corpus_utils import DfCorpus
from settings import DATASET_FOLDER

TWEET_ID = 'tweet_id'

TEST_CSV = 'mediaeval_coco_test.csv'
TEST_SIZE = 830

# orig. class labels
NO_MENTION = 1; MENTION = 2; SUPPORT = 3

# column names in the orig. csv files
CONSPIRACY_CAT_COLUMNS = [
    'class_label_for_Antivax_category',
    'class_label_for_Behaviour_and_Mind_Control_category',
    'class_label_for_Fake_virus_category',
    'class_label_for_Harmful_Radiation_Influence_category',
    'class_label_for_Intentional_Pandemic_category',
    'class_label_for_New_World_Order_category',
    'class_label_for_Population_reduction_Control_category',
    'class_label_for_Satanism_category',
    'class_label_for_Suppressed_cures_category'
]

# needs to be index aligned with the columns above
CONSPIRACY_CAT_IDS = [
    'Antivax',
    'Behaviour_and_Mind_Control',
    'Fake_virus',
    'Harmful_Radiation_Influence',
    'Intentional_Pandemic',
    'New_World_Order',
    'Population_reduction_Control',
    'Satanism',
    'Suppressed_cures'
]

# map cat ids to columns:
CONSPIRACY_CAT_ID2COL = { CONSPIRACY_CAT_IDS[i]: CONSPIRACY_CAT_COLUMNS[i] for i in range(len(CONSPIRACY_CAT_IDS)) }


def add_binary_category_classes(data: DataFrame, positive=[MENTION, SUPPORT]) -> DataFrame:
    '''
    For all the category columns, add a column with binary values
        (0/1) indicating whether the category value is among positive values or not.
    '''
    cats = CONSPIRACY_CAT_COLUMNS
    # copy cat columns to columns with '_orig' suffix
    for cat in cats: data[cat + '_orig'] = data[cat]
    for cat in cats:
        data[cat] = data[cat + '_orig'].apply(lambda v: 1 if int(v) in positive else 0)
    return data

def add_test_labels(df: pd.DataFrame):
    '''
    load labels for test dataset, form a file were columns are not named but the order
    is preserved, and add this data to the dataframe df with only tweet texts and ids.
    :return:
    '''
    # get column names (conspiracy category names) in the correct order from train dset
    categ_cols = ['tweet_id', 'class_label_for_Suppressed_cures_category', 'class_label_for_Behaviour_and_Mind_Control_category', 'class_label_for_Antivax_category', 'class_label_for_Fake_virus_category', 'class_label_for_Intentional_Pandemic_category', 'class_label_for_Harmful_Radiation_Influence_category', 'class_label_for_Population_reduction_Control_category', 'class_label_for_New_World_Order_category', 'class_label_for_Satanism_category', 'tweet_text']
    categ_cols = categ_cols[1:10]
    dset_file = Path(DATASET_FOLDER) / 'mediaeval_coco_test_labels.csv'
    col_names = [TWEET_ID] + categ_cols
    ldf = pd.read_csv(dset_file, sep=',', quoting=csv.QUOTE_NONE, names=col_names)
    assert set(df[TWEET_ID].tolist()) == set(ldf[TWEET_ID].tolist())
    rdf = pd.merge(df, ldf, on=TWEET_ID, how='inner')
    return rdf

def set_ids_as_columns(df: pd.DataFrame):
    '''
    Set category ids as column names, instead of original cat columns.
    '''
    for cat_id, cat_col in zip(CONSPIRACY_CAT_IDS, CONSPIRACY_CAT_COLUMNS):
        df[cat_id] = df[cat_col]
    df = df.drop(columns=CONSPIRACY_CAT_COLUMNS)
    return df

def test_dframe(labeled=True, binarize=False, ids_as_cols=False):
    dset_file = Path(DATASET_FOLDER) / TEST_CSV
    df = pd.read_csv(dset_file, sep=',', quoting=csv.QUOTE_NONE)
    assert len(df) == TEST_SIZE
    if labeled: df = add_test_labels(df)
    if binarize: df = add_binary_category_classes(df)
    if ids_as_cols: df = set_ids_as_columns(df)
    return df

def test_dframe_ment_noment():
    return test_dframe(labeled=True, binarize=True, ids_as_cols=True)

def test_corpus(binarize=False):
    df = test_dframe(binarize=binarize)
    return DfCorpus('mediaeval_test', df, 'tweet_id', 'tweet_text', properties=True)

def test_corpus_ment_noment():
    df = test_dframe_ment_noment()
    return DfCorpus('mediaeval_mentnoment_test', df, 'tweet_id', 'tweet_text', properties=True)


if __name__ == '__main__':
    pass