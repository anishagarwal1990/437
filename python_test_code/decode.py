#!/usr/bin/python

import numpy as np
import os

class Domain(object):
    def __init__(self, domain):
        self._domain = domain
        # Reverse key and values in dictionary
        self._lookup = dict([(v, i) for (i, v) in enumerate(domain)])
        self.m = len(domain)

    def text_to_seq(self, s):
        return [self._lookup[v] for v in s]

    def seq_to_text(self, x):
        return "".join([self._domain[i] for i in x])

class Cipher(object):
    def __init__(self, m, f=None):
        self.m = m
        if f is None:
            f = range(m)
        self.set_f(f)

    def set_f(self, f):
        self._f = f
        inv_dict = dict([(i, j) for (j, i) in enumerate(f)])
        self._inv_f = [inv_dict[i] for i in range(self.m)]

    def cipher(self, x):
        return [self._f[i] for i in x]

    def decipher(self, x):
        return [self._inv_f[i] for i in x]

    def swapx(self, i, j):
        r = self._f[i]
        self._f[i] = self._f[j]
        self._f[j] = r
        self.set_f(self._f)

class MCMCDecode(object):
    def __init__(self, domain, text_model, cipher, ciphertext):
        assert domain.m == text_model.m
        self.m = domain.m
        self._domain = domain
        self._text_model = text_model
        self.cipher = cipher
        self.load_ciphertext(ciphertext)
        self.trace_info = []
        self.count_proposal = 0
        self.count_acceptance = 0
        self._ll = 0.

    def load_ciphertext(self, ciphertext):
        self.ciphertext = ciphertext
        self.y = self._domain.text_to_seq(self.ciphertext)
        self.x = self.cipher.decipher(self.y)
        self.plaintext = self._domain.seq_to_text(self.x)
        self._ll = self._text_model.ll(self.x)

    def ll(self):
        return self._ll

    def MH(self, verbose=True):
        self.count_proposal += 1
        i, j = np.random.choice(range(self.m), 2, replace=False)
        ll_increment = self._text_model.ll_increment_swapx(self.x, i, j)
        MH_factor = np.exp(ll_increment)
        if np.random.rand() < MH_factor:
            self.count_acceptance += 1
            self._ll = self._ll + ll_increment
            self.cipher.swapx(i, j)
            x_proposed = np.array(self.x)
            x_proposed[x_proposed == i] = self.m
            x_proposed[x_proposed == j] = i
            x_proposed[x_proposed == self.m] = j
            x_proposed = list(x_proposed)
            self.x = x_proposed
            self.plaintext = self._domain.seq_to_text(self.x)
            accepted = True
        else:
            accepted = False
        if verbose:
            print "* Propose to swap %s: ll = %g, MH = %g, accepted = %s" % (
                self._domain.seq_to_text([i, j]), self._ll, MH_factor, accepted)
        return [accepted, ll_increment]

class textModel(object):
    def __init__(self, letter_probs, letter_transition):
        assert len(letter_probs) == letter_transition.shape[0] == letter_transition.shape[1]
        self.m = len(letter_probs)
        self.P = letter_probs
        self.T = letter_transition

    def ll(self, x):
        _ll = 0.
        for i, z in enumerate(x):
            if i == 0:
                _ll += np.log(self.P[z])
            else:
                _ll += np.log(self.T[z, x[i-1]])
        return _ll

    def ll_increment_swapx(self, x, a, b):
        def complement(v):
            if v == a:
                return b
            elif v == b:
                return a
            else:
                return v
        delta_ll = 0.
        for i, z in enumerate(x):
            if i == 0:
                z_ = complement(z)
                delta_ll += np.log(self.P[z_]) - np.log(self.P[z])
            else:
                u = x[i-1]
                if (z in (a, b)) or (u in (a, b)):
                    z_ = complement(z)
                    u_ = complement(u)
                    delta_ll += np.log(self.T[z_, u_]) - np.log(self.T[z, u])
        return delta_ll

data_path = os.path.dirname(os.path.abspath(__file__)) + "/"
with open(data_path + "alphabet.csv") as f_:
    domain = Domain(f_.read().rstrip().split(","))

letter_probs = np.genfromtxt(data_path + "letter_probabilities.csv", delimiter=",")
letter_transition = np.genfromtxt(data_path + "letter_transition_matrix.csv", delimiter=",")
letter_transition[letter_transition == 0.] = 1e-20
text_model = textModel(letter_probs, letter_transition)


def init_cipher(letter_probs, domain, ciphertext):
    m = domain.m
    y = domain.text_to_seq(ciphertext)
    idx_sorted_by_letter_probs = sorted(range(m), key=lambda k: letter_probs[k])
    count_ = np.zeros(m)
    for z in y:
        count_[z] += 1
    idx_sorted_by_ciphertext = sorted(range(m), key=lambda k: count_[k])
    f_ = [0] * m
    for a, b in zip(idx_sorted_by_letter_probs, idx_sorted_by_ciphertext):
        f_[a] = b

    return Cipher(m, f_)


def decode(ciphertext, output_file_name):
    max_iter=100000,
    max_no_improve=1000
    verbose=True
    cipher = init_cipher(letter_probs, domain, ciphertext)
    mcmcdecoder = MCMCDecode(domain, text_model, cipher, ciphertext)
    ll_non_increase_cnt = 0
    ll = 0.
    for iter in range(max_iter):
        mcmcdecoder.MH()
        ll_ = mcmcdecoder.ll()
        if ll_ > ll:
            ll = ll_
            ll_non_increase_cnt = 0
        else:
            ll_non_increase_cnt += 1
        if ll_non_increase_cnt > max_no_improve:
            if verbose:
                print "* ended with %d rounds of no improve" % (ll_non_increase_cnt)
            break
        if verbose and iter % 500 == 1:
            print mcmcdecoder.plaintext[0:200]
            print "iter = ", iter, "ll = ", mcmcdecoder.ll()
    fw = open(output_file_name, "w")
    fw.write(mcmcdecoder.plaintext)
    fw.close()
    return 0



