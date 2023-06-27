from abc import abstractmethod, ABC

import pandas as pd

class Identifiable(ABC):

    @property
    @abstractmethod
    def id(self): ...


class Corpus(Identifiable):

    @property
    @abstractmethod
    def id(self): ...

    @abstractmethod
    def get_text(self, id): ...

    def __getitem__(self, id):
        return self.get_text(id)

    def get_texts(self, ids):
        return [self.get_text(id) for id in ids]

    @abstractmethod
    def __iter__(self): ...

    def text_ids(self):
        return [ txto.id for txto in self ]

    def __len__(self):
        if not hasattr(self, '_length'):
            self._length = sum(1 for _ in self)
        return self._length

    def __contains__(self, txt_id):
        return txt_id in self.text_ids()


class DfCorpus(Corpus):
    '''
    Implementation of a corpus with texts (and related data) residing in a Pandas DataFrame
    '''

    def __init__(self, cid, df : pd.DataFrame, id_col, txt_col, properties=None):
        self._id = cid
        self._df = df
        self._idc, self._txtc = id_col, txt_col
        self._df.set_index(id_col, verify_integrity=True, inplace=True) # verify id uniqueness
        self._process_props(properties)

    def _process_props(self, props):
        if not props:
            self._props = None
        else:
            cols = [c for c in self._df if c not in [self._idc, self._txtc]]
            if props == True: self._props = cols # add all
            else:
                self._props = []
                for prop in props:
                    if prop in cols: self._props.append(prop)
                    else: raise ValueError(f'Property {prop} is not a column of a dataframe.')

    @property
    def id(self): return self._id

    def _get_text_props(self, id):
        if self._props is None: return {}
        else:
            return { p: self._df.loc[id, p] for p in self._props }

    def get_text(self, id):
        if id in self._df.index:
            strtxt = self._df.loc[id, self._txtc]
            props = self._get_text_props(id)
            return SimpleText(id, strtxt, **props)
        else: return None

    def get_texts(self, ids):
        return [self.get_text(id) for id in ids]

    def __iter__(self):
        for id in self._df.index:
            yield self.get_text(id)

    def text_ids(self):
        return list(self._df.index)

    def __contains__(self, txt_id):
        return txt_id in self._df.index

    def __len__(self):
       return len(self._df)


class Text(Identifiable):

    @property
    @abstractmethod
    def text(self): ...

    def __str__(self): return self.text

    @abstractmethod
    def __iter__(self): ...

class SimpleText(Text):

    @property
    def id(self): return self._id

    @property
    def text(self): return self._text

    def __init__(self, id, text, **attributes):
        self._id = id
        self._text = text
        for attr, val in attributes.items():
            self.__dict__[attr] = val

    def __iter__(self):
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if key != 'id' and key != 'text':
                    yield key, value

def copy_text(txt):
    atts = { name:val for name, val in txt }
    return Text(txt.id, txt.text, **atts)
