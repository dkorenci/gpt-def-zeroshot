from typing import List

from scipy.spatial.distance import cosine, cdist
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

from experiments.data_utils import CONSPIRACY_CAT_IDS
from experiments.definition_utils import load_all_gen_defs, load_all_def_scores, \
    golddefs_as_catmap, gendef_res_coprediction_matrix_4seeds


emb_models = {}
def get_emb_model(model_name='all-mpnet-base-v2'):
    global emb_models
    if model_name not in emb_models:
        emb_models[model_name] = SentenceTransformer(model_name)
    return emb_models[model_name]

def embed_texts(texts: List[str], emb_model='all-mpnet-base-v2'):
    st = get_emb_model(emb_model)
    return st.encode(texts)

def calc_gendef_golddef_correlation(seed_cat_def_score, cat2gold_emb, emb_model=emb_models):
    gen_defs = [d for _, _, d, _ in seed_cat_def_score]
    gen_defs_embs = embed_texts(gen_defs, emb_model=emb_model)
    cos2gold = []
    for i, (seed, cat_id, gen_def, scores) in enumerate(seed_cat_def_score):
        cos2gold.append(cosine(gen_defs_embs[i], cat2gold_emb[cat_id]))
    for score_fns in ['_MCC', 'F1', 'R', 'P']:
        scores = []
        for _, cat_id, _, s in seed_cat_def_score:
            scr = s[score_fns][CONSPIRACY_CAT_IDS.index(cat_id)]
            scores.append(scr)
        spr = spearmanr(scores, cos2gold)
        print(f'Spearman corr. between {score_fns} and [gen2gold] dist: {spr.correlation:3.3f}, p {spr.pvalue:3.3f};')

def gendefs_golddist2scores(emb_model='all-mpnet-base-v2'):
    '''
    The correlations use cosine distance so they need to be inverted to get correlations with similarity.
    '''
    # prepare data
    SEEDS = [0, 1, 2, 3, 4]
    gen_defs = load_all_gen_defs() # map seed -> { cat_id -> def }
    gen_def_scores = load_all_def_scores() # map seed -> MultiCategoryScorer
    gold_defs = golddefs_as_catmap() # map cat_id -> gold def
    gold_defs_embs = embed_texts([gold_defs[c] for c in CONSPIRACY_CAT_IDS], emb_model=emb_model)
    cat2gold_emb = {c: gold_defs_embs[i] for i, c in enumerate(CONSPIRACY_CAT_IDS)}
    # flatten the results, to get (seed, cat, def, score) tuples
    seed_cat_def_score = []
    for seed in SEEDS:
        for cat_id in gen_defs[seed]:
            _, scores = gen_def_scores[seed].score()
            seed_cat_def_score.append((seed, cat_id, gen_defs[seed][cat_id], scores))
    print('GLOBAL CORRELATION')
    calc_gendef_golddef_correlation(seed_cat_def_score, cat2gold_emb, emb_model=emb_model)
    # for seed in SEEDS:
    #     seed_cat_def_score_seed = [s for s in seed_cat_def_score if s[0] == seed]
    #     assert(len(seed_cat_def_score_seed) == 9) # nine categories
    #     assert(set([s[1] for s in seed_cat_def_score_seed]) == set(CONSPIRACY_CAT_IDS)) # all categories covered
    #     print(f'CORRELATION FOR SEED {seed}')
    #     calc_gendef_golddef_correlation(seed_cat_def_score_seed, cat2gold_emb, emb_model=emb_model)

def calc_gendef_coprediction(seed_cat_def_score, emb_model, seeds):
    gen_defs = [d for _, _, d, _ in seed_cat_def_score]
    gen_defs_embs = embed_texts(gen_defs, emb_model=emb_model)
    gen_def_dist = cdist(gen_defs_embs, gen_defs_embs, metric='cosine')
    # calculate correlation
    copreds = []
    dists = []
    # cache results of gendef_res_coprediction_matrix, for each pair of seeds
    copred_cached = {}
    for s1 in seeds:
        for s2 in seeds:
            if s1 <= s2:
                copred_cached[(s1, s2)] = gendef_res_coprediction_matrix_4seeds(s1, s2)
            else:
                copred_cached[(s1, s2)] = copred_cached[(s2, s1)]
    for i, (seed1, cat_id1, gen_def1, scores1) in enumerate(seed_cat_def_score):
        for j, (seed2, cat_id2, gen_def2, scores2) in enumerate(seed_cat_def_score):
            if i < j:
                copred = copred_cached[(seed1, seed2)]
                copreds.append(copred[cat_id1][cat_id2])
                dists.append(gen_def_dist[i][j])
    # correlation between copreds and dists
    spr = spearmanr(copreds, dists)
    print(f'Spearman corr. between coprediction and sem.emb dists: {spr.correlation:3.3f}, p {spr.pvalue:3.3f};')

def gendef_coprediction(emb_model='all-mpnet-base-v2'):
    '''
    The correlations use cosine distance so they need to be inverted to get correlations with similarity.
    '''
    # prepare data
    SEEDS = [0, 1, 2, 3, 4]
    gen_defs = load_all_gen_defs() # map seed -> { cat_id -> def }
    gen_def_scores = load_all_def_scores() # map seed -> MultiCategoryScorer
    gold_defs = golddefs_as_catmap() # map cat_id -> gold def
    # convert gold defs to embeddings
    # gold_defs_embs = embed_texts([gold_defs[c] for c in CONSPIRACY_CAT_IDS], emb_model=emb_model)
    # cat2gold_emb = {c: gold_defs_embs[i] for i, c in enumerate(CONSPIRACY_CAT_IDS)}
    seed_cat_def_score = []
    for seed in SEEDS:
        for cat_id in gen_defs[seed]:
            _, scores = gen_def_scores[seed].score()
            seed_cat_def_score.append((seed, cat_id, gen_defs[seed][cat_id], scores))
    print('GLOBAL CORRELATION')
    calc_gendef_coprediction(seed_cat_def_score, emb_model=emb_model, seeds=SEEDS)
    # for seed in SEEDS:
    #     seed_cat_def_score_seed = [s for s in seed_cat_def_score if s[0] == seed]
    #     assert(len(seed_cat_def_score_seed) == 9) # nine categories
    #     assert(set([s[1] for s in seed_cat_def_score_seed]) == set(CONSPIRACY_CAT_IDS)) # all categories covered
    #     print(f'CORRELATION FOR SEED {seed}')
    #     calc_gendef_coprediction(seed_cat_def_score_seed, emb_model=emb_model, seeds=SEEDS)


if __name__ == '__main__':
    # The correlations use cosine distance so they need to be inverted to get correlations with similarity.
    gendefs_golddist2scores()
    gendef_coprediction()


