import numpy as np
import pandas as pd
import pickle
import re
import usaddress

from typing import List, Tuple, Union

from address_normalize.paths import DATA_DIR


def address_parser(address: str) -> Tuple[str, str, str]:
    """
    Parses out address number and street name, all lowercase"""
    address = address.strip().lower()
    try:
        parsed = usaddress.tag(address)[0]
        address_number = parsed['AddressNumber']
        if 'AddressNumber' not in parsed:
            address_number = ''
    except:
        # hack for now, doesn't really work, but will for my purpose
        try:
            address_number, street_name = re.match(r'(\d+)(.*)', address).groups()
        except:
            address_number = ''
            street_name = address

    street_name = address.replace(address_number, '')
    return address, address_number.strip(), street_name.strip()


def construct_corpus(corpus_sents: List[str]):
    corpus_dict = {}

    for address in corpus_sents:
        if address.strip() != '':
            an, street_name = address_parser(address)
            if an in corpus_dict:
                if street_name not in corpus_dict:
                    corpus_dict[an]['street_names'].append(street_name)
                    corpus_dict[an]['full_address'].append(address)

            else:
                corpus_dict[an] = {}
                corpus_dict[an]['street_names'] = [street_name]
                corpus_dict[an]['full_address'] = [address]

    return corpus_dict


def save_corpus(corpus_dict: dict, directory: str):
    with open(directory, 'wb') as fp:
        pickle.dump(corpus_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_corpus(directory: str):
    with open(directory, 'rb') as fp:
        return pickle.load(fp)


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

