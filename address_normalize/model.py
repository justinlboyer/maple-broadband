import numpy as np

from typing import List, Union

from address_normalize.paths import DATA_DIR
from address_normalize.utils import address_parser, jaccard_similarity, load_corpus


class Model:
    def __init__(self) -> None:
        self.corpus_dict = load_corpus(DATA_DIR / 'corpus_dict.pkl')

    def __call__(self, queries: List[str]) -> List[Union[str, float, str]]:
        """
        Return the address, score, and top match"""
        data = []
        for address in queries:

            query_address, query_address_number, query_street_names = address_parser(address)

            try:
                filtered_corpus = self.corpus_dict[query_address_number]
            except KeyError:
                print(f"No address number match for {query_address}")
            else:
                scores = []
                for street_name in filtered_corpus['street_names']:
                    scores.append(jaccard_similarity(query_street_names.split(), street_name.split()))

                scores = np.array(scores)
                top_arg = scores.argmax()
                row = [query_address, scores[top_arg], filtered_corpus['full_address'][top_arg]]

                data.append(row)

        return data
