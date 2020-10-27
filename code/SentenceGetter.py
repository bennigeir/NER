# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:01:09 2020

@author: Benedikt
"""

# Data needs to be parsed into sentences
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p) for w, p in zip(s['Token'].values.tolist(),
                                                     s['Tag'].values.tolist())]
        self.grouped = self.data.groupby("Sentence no.").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None