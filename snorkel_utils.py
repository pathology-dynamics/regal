from snorkel.labeling import LabelingFunction
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.labeling import LFAnalysis
from snorkel.labeling import PandasLFApplier
from snorkel.preprocess import preprocessor


ABSTAIN = -1

def keyword_lookup(x, keyword, label):
    '''
    Labeling function for substring lookup
    '''
    if keyword in x:
        return label
    # if type(keywords) == str:
    #     keywords = [keywords]
    # if any(word in x for word in keywords):
    #     return label
    return ABSTAIN


def make_keyword_lf(keyword, label, rpn_generated=True):
    if type(keyword) == list:
        raise ValueError("Keyword should be a single word!")
        # keywords = [keywords
    if rpn_generated == True:
        name = f"rpn_keyword_{keyword}"
    else:
        name = f"keyword_{keyword}"
    return LabelingFunction(
        name=name,
        f=keyword_lookup,
        resources=dict(keyword=keyword, label=label),
    )


def token_id_lookup(input_ids, rule_ids, label):
    '''
    Labeling function for token id lookup
    '''
    if type(rule_ids) not in [int, list]:
        raise ValueError("Token type must be int or list")
    if type(rule_ids) == int:
        rule_ids = [rule_ids]

    if any(rule_id in input_ids for rule_id in rule_ids):
        return label

    return ABSTAIN

def make_token_lf(rule_ids, label, rpn_generated=True):
    if rpn_generated == True:
        name = f"rpn_token_{rule_ids[0]}"
    else:
        name = f"token_{rule_ids[0]}"
    return LabelingFunction(
        name=name,
        f=token_id_lookup,
        resources=dict(rule_ids=rule_ids, label=label),
    )