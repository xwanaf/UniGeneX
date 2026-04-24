from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional

# from torchtext._torchtext import Vocab as VocabPybind

from .vocab import Vocab

class VocabPybind:
    def __init__(self, tokens: List[str], default_index: Optional[int] = None):
        """
        Equivalent to Vocab::Vocab(StringList tokens, const c10::optional<int64_t>& default_index)
        """
        self.itos_ = []
        self.stoi_ = {}  # Python dict replaces the C++ fixed-size vector/hash map
        self.default_index_ = default_index
        self.version_str_ = "0.0.2"

        for token in tokens:
            # TORCH_CHECK logic for duplicate tokens
            if token in self.stoi_:
                raise RuntimeError(f"Duplicate token found in tokens list: {token}")
            self._add(token)

    def _add(self, token: str) -> None:
        """Internal helper to mimic the C++ _add method."""
        self.stoi_[token] = len(self.itos_)
        self.itos_.append(token)

    def __len__(self) -> int:
        # int64_t Vocab::__len__()
        return len(self.itos_)

    def __contains__(self, token: str) -> bool:
        # bool Vocab::__contains__
        return token in self.stoi_

    def __getitem__(self, token: str) -> int:
        # int64_t Vocab::__getitem__
        if token in self.stoi_:
            return self.stoi_[token]

        if self.default_index_ is not None:
            return self.default_index_

        raise RuntimeError(
            f"Token {token} not found and default index is not set"
        )

    def set_default_index(self, index: Optional[int]) -> None:
        # void Vocab::set_default_index
        self.default_index_ = index

    def get_default_index(self) -> Optional[int]:
        # c10::optional<int64_t> Vocab::get_default_index
        return self.default_index_

    def append_token(self, token: str) -> None:
        # void Vocab::append_token
        if token in self.stoi_:
            raise RuntimeError(
                f"Token {token} already exists in the Vocab with index: {self.stoi_[token]}"
            )
        self._add(token)

    def insert_token(self, token: str, index: int) -> None:
        # void Vocab::insert_token
        if not (0 <= index <= len(self.itos_)):
            raise RuntimeError(
                f"Specified index {index} is out of bounds for vocab of size {len(self.itos_)}"
            )

        if self.__contains__(token):
            raise RuntimeError(f"Token {token} already exists in Vocab")

        # Offset tokens logic from C++
        self.itos_.insert(index, token)
        # Rebuild stoi_ map to reflect new indices (equivalent to the C++ loop)
        self.stoi_ = {t: i for i, t in enumerate(self.itos_)}

    def lookup_token(self, index: int) -> str:
        # std::string Vocab::lookup_token
        if not (0 <= index < len(self.itos_)):
            raise RuntimeError(
                f"Specified index {index} is out of bounds for vocab of size {len(self.itos_)}"
            )
        return self.itos_[index]

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        # StringList Vocab::lookup_tokens
        tokens = []
        for i, idx in enumerate(indices):
            if not (0 <= idx < len(self.itos_)):
                raise RuntimeError(
                    f"Specified index {idx} at position {i} is out of bounds for vocab of size {len(self.itos_)}"
                )
            tokens.append(self.itos_[idx])
        return tokens

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        # std::vector<int64_t> Vocab::lookup_indices
        return [self.__getitem__(token) for token in tokens]

    def get_stoi(self) -> Dict[str, int]:
        # std::unordered_map<std::string, int64_t> Vocab::get_stoi
        return self.stoi_.copy()

    def get_itos(self) -> List[str]:
        # StringList Vocab::get_itos
        return self.itos_.copy()
    
    
def vocab(
    ordered_dict: Dict, min_freq: int = 1, specials: Optional[List[str]] = None, special_first: bool = True
) -> Vocab:
    r"""Factory method for creating a vocab object which maps tokens to indices.

    Note that the ordering in which key value pairs were inserted in the `ordered_dict` will be respected when building the vocab.
    Therefore if sorting by token frequency is important to the user, the `ordered_dict` should be created in a way to reflect this.

    Args:
        ordered_dict: Ordered Dictionary mapping tokens to their corresponding occurance frequencies.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
        specials: Special symbols to add. The order of supplied tokens will be preserved.
        special_first: Indicates whether to insert symbols at the beginning or at the end.

    Returns:
        torchtext.vocab.Vocab: A `Vocab` object

    Examples:
        >>> from torchtext.vocab import vocab
        >>> from collections import Counter, OrderedDict
        >>> counter = Counter(["a", "a", "b", "b", "b"])
        >>> sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        >>> ordered_dict = OrderedDict(sorted_by_freq_tuples)
        >>> v1 = vocab(ordered_dict)
        >>> print(v1['a']) #prints 1
        >>> print(v1['out of vocab']) #raise RuntimeError since default index is not set
        >>> tokens = ['e', 'd', 'c', 'b', 'a']
        >>> #adding <unk> token and default index
        >>> unk_token = '<unk>'
        >>> default_index = -1
        >>> v2 = vocab(OrderedDict([(token, 1) for token in tokens]), specials=[unk_token])
        >>> v2.set_default_index(default_index)
        >>> print(v2['<unk>']) #prints 0
        >>> print(v2['out of vocab']) #prints -1
        >>> #make default index same as index of unk_token
        >>> v2.set_default_index(v2[unk_token])
        >>> v2['out of vocab'] is v2[unk_token] #prints True
    """
    specials = specials or []
    for token in specials:
        ordered_dict.pop(token, None)

    tokens = []
    # Save room for special tokens
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append(token)

    if special_first:
        tokens[0:0] = specials
    else:
        tokens.extend(specials)

    return Vocab(VocabPybind(tokens, None))



def build_vocab_from_iterator(
    iterator: Iterable,
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
    max_tokens: Optional[int] = None,
) -> Vocab:
    """
    Build a Vocab from an iterator.

    Args:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
        specials: Special symbols to add. The order of supplied tokens will be preserved.
        special_first: Indicates whether to insert symbols at the beginning or at the end.
        max_tokens: If provided, creates the vocab from the `max_tokens - len(specials)` most frequent tokens.


    Returns:
        torchtext.vocab.Vocab: A `Vocab` object

    Examples:
        >>> #generating vocab from text file
        >>> import io
        >>> from torchtext.vocab import build_vocab_from_iterator
        >>> def yield_tokens(file_path):
        >>>     with io.open(file_path, encoding = 'utf-8') as f:
        >>>         for line in f:
        >>>             yield line.strip().split()
        >>> vocab = build_vocab_from_iterator(yield_tokens(file_path), specials=["<unk>"])
    """

    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    specials = specials or []

    # First sort by descending frequency, then lexicographically
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    if max_tokens is None:
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
    else:
        assert len(specials) < max_tokens, "len(specials) >= max_tokens, so the vocab will be entirely special tokens."
        ordered_dict = OrderedDict(sorted_by_freq_tuples[: max_tokens - len(specials)])

    word_vocab = vocab(ordered_dict, min_freq=min_freq, specials=specials, special_first=special_first)
    return word_vocab
