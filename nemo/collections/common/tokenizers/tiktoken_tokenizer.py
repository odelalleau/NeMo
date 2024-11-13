# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tiktoken
except ImportError:
    pass

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = ['TiktokenTokenizer']


def reload_mergeable_ranks(
    path: str,
    max_vocab: Optional[int] = None,
) -> Dict[bytes, int]:
    """
    Reload the tokenizer JSON file and convert it to Tiktoken format.
    """
    assert path.endswith(".json")

    # reload vocab
    with open(path, "r") as f:
        vocab = json.load(f)
    assert isinstance(vocab, list)
    print(f"Vocab size: {len(vocab)}")
    if max_vocab is not None:
        vocab = vocab[:max_vocab]
        print(f"Cutting vocab to first {len(vocab)} tokens.")

    # build ranks
    ranks: Dict[bytes, int] = {}
    for i, x in enumerate(vocab):
        assert x.keys() == {"rank", "token_bytes", "token_str"}
        assert x["rank"] == i
        merge = base64.b64decode(x["token_bytes"])
        assert i >= 256 or merge == bytes([i])
        ranks[merge] = x["rank"]

    # sanity check
    assert len(ranks) == len(vocab)
    assert set(ranks.values()) == set(range(len(ranks)))

    return ranks


PATTERN_TIKTOKEN = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
DEFAULT_TIKTOKEN_MAX_VOCAB = 2**17  # 131072
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>"]
SPECIAL_TOKEN_TEMPLATE = "<SPECIAL_{id}>"


class TiktokenTokenizer(TokenizerSpec):
    """
    TiktokenTokenizer https://github.com/openai/tiktoken.

    Args:
        model_path: path to tokenizer vocabulary
        num_special_tokens: number of special tokens to generate
        special_tokens: template for user-defined special tokens
        pattern: Regex pattern to split the text
    """

    def __init__(
        self,
        vocab_file: str,
        pattern: str = PATTERN_TIKTOKEN,
        vocab_size: int = DEFAULT_TIKTOKEN_MAX_VOCAB,  # 131072
        num_special_tokens: int = 1000,
        special_tokens: Optional[List[str]] = None,
    ):
        if not vocab_file or not os.path.exists(vocab_file):
            raise ValueError(f"vocab_file: {vocab_file} is invalid")

        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS.copy()

        assert len(special_tokens) == len(set(special_tokens)), f"Special tokens should be unique: {special_tokens}"
        assert len(special_tokens) <= num_special_tokens < vocab_size
        assert set(SPECIAL_TOKENS) <= set(special_tokens), f"Custom special tokens should include {SPECIAL_TOKENS}"

        self._unk_id = special_tokens.index("<unk>")
        self._bos_id = special_tokens.index("<s>")
        self._eos_id = special_tokens.index("</s>")

        self._vocab_size = vocab_size

        self.num_special_tokens = num_special_tokens
        special_filler = [SPECIAL_TOKEN_TEMPLATE.format(id=i) for i in range(len(special_tokens), num_special_tokens)]
        if special_filler:
            print(f"Adding special tokens {special_filler[0]}, ..., {special_filler[-1]}")
        self.special_tokens = special_tokens + special_filler
        assert len(set(self.special_tokens)) == len(self.special_tokens) == num_special_tokens, self.special_tokens
        self.inner_vocab_size = vocab_size - num_special_tokens

        # reload vocab
        self.token2id = reload_mergeable_ranks(vocab_file, max_vocab=self.inner_vocab_size)
        self.id2token = {v: k for k, v in self.token2id.items()}
        assert set(range(self.inner_vocab_size)) == set(self.id2token.keys())

        self.shifted_id2token = {i: tok for i, tok in enumerate(self.special_tokens)}
        for key, value in self.id2token.items():
            self.shifted_id2token[key + self.num_special_tokens] = value

        self.tokenizer = tiktoken.Encoding(
            name=Path(vocab_file).parent.name,
            pat_str=pattern,
            mergeable_ranks=self.token2id,
            special_tokens={},  # special tokens are handled manually
        )

    def text_to_tokens(self, text: str):
        tokens = []
        special_token_pattern = SPECIAL_TOKEN_TEMPLATE.format(id='\\d+')
        parts = re.split(f"({special_token_pattern})", text)
        for part in parts:
            if re.match(special_token_pattern, part):
                tokens.append(part.encode('utf-8'))
            else:
                token_ids = self.tokenizer.encode(part)
                tokens.extend([self.tokenizer.decode_single_token_bytes(token) for token in token_ids])
        return tokens

    def tokens_to_text(self, tokens: List[int]):
        result = []
        for token in tokens:
            if isinstance(token, bytes):
                result.append(token.decode('utf-8'))
            else:
                result.append(self.tokenizer.decode([token]))
        return ''.join(result)

    def token_to_id(self, token):
        token_str = token.decode('utf-8', errors='replace') if isinstance(token, bytes) else token
        if token_str in self.special_tokens:
            return self.special_tokens.index(token_str)
        else:
            token_ids = self.tokenizer.encode(token_str)
            if len(token_ids) != 1:
                raise ValueError(f"Token '{token_str}' should correspond to exactly one ID, but got {token_ids}")
            return token_ids[0] + self.num_special_tokens

    def tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            token_str = token.decode('utf-8', errors='replace') if isinstance(token, bytes) else token
            if token_str in self.special_tokens:
                ids.append(self.special_tokens.index(token_str))
            else:
                ids.extend([id + self.num_special_tokens for id in self.tokenizer.encode(token_str)])
        return ids

    def ids_to_tokens(self, token_ids):
        tokens = []
        for token_id in token_ids:
            if token_id < self.num_special_tokens:
                tokens.append(self.special_tokens[token_id].encode('utf-8'))
            else:
                adjusted_token = token_id - self.num_special_tokens
                token_bytes = self.tokenizer.decode_single_token_bytes(adjusted_token)
                tokens.append(token_bytes)
        return tokens

    def text_to_ids(self, text: str):
        tokens = []
        special_token_pattern = SPECIAL_TOKEN_TEMPLATE.format(id='\\d+')
        parts = re.split(f"({special_token_pattern})", text)
        for part in parts:
            if re.match(special_token_pattern, part):
                token_id = int(re.findall(r"\d+", part)[0])
                tokens.append(token_id)
            else:
                token_ids = self.tokenizer.encode(part)
                tokens.extend([t + self.num_special_tokens for t in token_ids])
        return tokens

    def ids_to_text(self, tokens: List[int], skip_special_tokens: bool = True):
        result = []
        for token in tokens:
            if token < self.num_special_tokens:
                if not skip_special_tokens:
                    result.append(self.special_tokens[token])
            else:
                adjusted_token = token - self.num_special_tokens
                result.append(self.tokenizer.decode([adjusted_token]))
        return ''.join(result)

    @property
    def bos_id(self):
        return self._bos_id

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def vocab(self):
        return self.token2id

    @property
    def decoder(self):
        return self.shifted_id2token

    @property
    def encoder(self):
        return self.vocab

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
