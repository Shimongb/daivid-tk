import numpy as np
import title as tdt


def get_item_by_score(items, list_len=1):
    return [items[i] for i in np.random.choice(len(items), list_len, replace=True, p=[i['score'] for i in items])]


def get_prime_str(lst_pre, lst_post, join_char=''):
    return get_item_by_score(lst_pre)[0]['text'] + ' {}'.format(join_char) + get_item_by_score(lst_post)[0]['text']


def get_title(pre_list=tdt.pre_list, post_list=tdt.post_list, join_char=tdt.join_char):
    return get_prime_str(pre_list, post_list, join_char)
