import torch


def index_if_list_else_nbr(nbr_or_list, index):
    if isinstance(nbr_or_list, int):
        return nbr_or_list
    else:
        return nbr_or_list[index]
    

