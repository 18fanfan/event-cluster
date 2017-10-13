import numpy as np


def dcg(rank_array):
    gain = rank_array
    discounts = np.log2(np.arange(rank_array.size) + 2)
    return np.sum(gain / discounts)

def exp_dcg(rank_array):
    gain = 2 ** rank_array - 1
    discounts = np.log2(np.arange(rank_array.size) + 2)
    return np.sum(gain / discounts)


def lb_ndcg(rank_array, dcg_func=dcg):
    # the list contain relavent score
    # we are not going to use 2**rel score duo to the score is smaller than 1
    cdcg = dcg_func(rank_array)
    # idcg
    upper_bound = dcg_func(np.sort(rank_array)[::-1])
    lower_bound = dcg_func(np.sort(rank_array))
    print upper_bound, lower_bound, cdcg
    return (cdcg-lower_bound) / (upper_bound-lower_bound)

def test_lb_ndcg(a):
    print a
    print lb_ndcg(a)
    print lb_ndcg(a, dcg_func=exp_dcg)
    print


#rank_array = np.array([0.3, 0.5, 0.2, 0, 0], dtype=float)
#print lb_ndcg(rank_array)
test_lb_ndcg(np.array([0.5, 0.3, 0.2, 0, 0], dtype=float))
test_lb_ndcg(np.array([0.5, 0.1, 0.3, 0.2, 0, 0], dtype=float))
test_lb_ndcg(np.array([0.0, 0.5, 0.3, 0.2, 0, 0], dtype=float))
test_lb_ndcg(np.array([0.5, 0.0, 0.3, 0.2, 0, 0], dtype=float))
test_lb_ndcg(np.array([0.5, 0.3, 0.0, 0.2, 0, 0], dtype=float))
test_lb_ndcg(np.array([0.5, 0.3, 0.2, 0.0, 0, 0], dtype=float))
test_lb_ndcg(np.array([0, 0, 0.2, 0.3, 0.5], dtype=float))

    


