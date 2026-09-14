from typing import List, Optional, OrderedDict, Tuple, Union
import orjson

def load_dataset(filepaths: Union[str, List[str]]) -> List[dict]:
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    dataset = []
    for filepath in filepaths:
        with open(filepath, encoding="utf-8") as datas:
            dataset += [orjson.loads(d) for d in datas]

    return dataset