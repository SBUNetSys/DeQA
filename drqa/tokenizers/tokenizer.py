#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Base tokenizer/tokens classes and utilities."""

import copy


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    CHAR = 1
    TEXT_WS = 2
    SPAN = 3
    POS = 4
    LEMMA = 5
    NER = 6

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def chars(self, uncased=False):
        """Returns a list of the first character of each token

        Args:
            uncased: lower cases characters
        """
        if uncased:
            return [t[self.CHAR].lower() for t in self.data]
        else:
            return [t[self.CHAR] for t in self.data]

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_strings: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups

    def __repr__(self):
        return ' '.join(self.words()).strip()


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    NER = ['O', 'MISC', 'PERSON', 'LOCATION', 'DATE', 'NUMBER', 'ORGANIZATION',
           'SET', 'DURATION', 'ORDINAL', 'PERCENT', 'MONEY', 'TIME', ]

    POS = ['RB', ',', 'DT', 'NN', 'VBZ', 'JJ', '.', 'IN', 'NNP', 'POS', 'CC', 'VBG',
           'PRP', 'NNS', 'VBN', '``', 'NNPS', "''", 'TO', 'WRB', 'VBD', 'CD', '-LRB-',
           'WDT', '-RRB-', 'RBS', 'VBP', 'VB', 'JJS', ':', 'PRP$', 'WP', 'JJR', '$', 'RP',
           'MD', 'EX', '#', 'RBR', 'FW', 'WP$', 'UH', 'PDT', 'SYM', 'LS']
    FEAT = NER + POS
    NER_DICT = {f: i for i, f in enumerate(NER)}
    POS_DICT = {f: i for i, f in enumerate(POS)}
    FEAT_DICT = {f: i for i, f in enumerate(FEAT)}

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()
