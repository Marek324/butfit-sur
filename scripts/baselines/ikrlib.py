"""
ikrlib.py - course-provided utility library, copied verbatim from the SUR
syllabus. Used here only to evaluate the teacher's baseline systems against
our production pipeline. Not part of the submitted system.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d
from matplotlib.pyplot import imread
from glob import glob
from math import log
import numpy as np
from numpy import ravel, diag, newaxis, ones, zeros, array, vstack, hstack, dstack, pi, tile
from numpy.linalg import eigh, det, inv, solve, norm
from numpy.random import rand, randn, randint
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.linalg import eigh as eigh_
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import scipy.fftpack


def k_nearest_neighbours(test_data, class1, class2, k):
    euclidean = cdist(np.r_[class1, class2], test_data)
    i = np.argsort(euclidean.T)
    return np.sum(i[:,:k] >= len(class1), axis=1) / float(k)


def gellipse(mu, cov, n=100, *args, **kwargs):
    mu = mu.copy()
    if mu.shape == (2,):
        mu.shape = (2, 1)
    if mu.shape != (2, 1) or cov.shape != (2, 2):
        raise RuntimeError('mu must be a two element vector and cov must be 2 x 2 matrix')
    d, v = eigh(4 * cov)
    d = diag(d)
    t = np.linspace(0, 2 * pi, n)
    x = v.dot(np.sign(d)).dot(np.sqrt(np.abs(d))).dot(array([np.cos(t), np.sin(t)])) + mu
    return plt.plot(x[0], x[1], *args, **kwargs)


def plot2dfun(f, limits, resolution, ax=None):
    if ax is None:
        ax = plt
    xmin, xmax, ymin, ymax = limits
    xlim = np.arange(ymin, ymax, (ymax - ymin) / float(resolution))
    ylim = np.arange(xmin, xmax, (xmax - xmin) / float(resolution))
    (b, a) = np.meshgrid(xlim, ylim)
    return ax.imshow(f(vstack([ravel(a), ravel(b)]).T).reshape(b.shape).T[::-1, :],
                     cmap='gray', aspect='auto', extent=(xmin, xmax, ymin, ymax))


def logpdf_gauss(x, mu, cov):
    assert(mu.ndim == 1 and len(mu) == len(cov) and (cov.ndim == 1 or cov.shape[0] == cov.shape[1]))
    x = np.atleast_2d(x) - mu
    if cov.ndim == 1:
        return -0.5*(len(mu)*np.log(2 * pi) + np.sum(np.log(cov)) + np.sum((x**2)/cov, axis=1))
    else:
        return -0.5*(len(mu)*np.log(2 * pi) + np.linalg.slogdet(cov)[1] + np.sum(x.dot(inv(cov)) * x, axis=1))


def train_gauss(x):
    return np.mean(x, axis=0), np.cov(x.T, bias=True)


def rand_gauss(n, mu, cov):
    if cov.ndim == 1:
        cov = np.diag(cov)
    assert(mu.ndim == 1 and len(mu) == len(cov) and cov.ndim == 2 and cov.shape[0] == cov.shape[1])
    d, v = eigh(cov)
    return (randn(n, len(mu)) * np.sqrt(d)).dot(v) + mu


def logpdf_gmm(x, ws, mus, covs):
    return logsumexp([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)], axis=0)


def train_gmm(x, ws, mus, covs):
    gamma = np.vstack([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)])
    logevidence = logsumexp(gamma, axis=0)
    gamma = np.exp(gamma - logevidence)
    tll = logevidence.sum()
    gammasum = gamma.sum(axis=1)
    ws = gammasum / len(x)
    mus = gamma.dot(x)/gammasum[:,np.newaxis]
    if covs[0].ndim == 1:
        covs = gamma.dot(x**2)/gammasum[:,np.newaxis] - mus**2
    else:
        covs = np.array([(gamma[i]*x.T).dot(x)/gammasum[i] - mus[i][:, newaxis].dot(mus[[i]]) for i in range(len(ws))])
    return ws, mus, covs, tll


def train_generative_linear_classifier(x, class_id):
    true_data  = x[class_id == 1]
    false_data = x[class_id != 1]
    (true_mean, true_cov) = train_gauss(true_data)
    (false_mean, false_cov) = train_gauss(false_data)
    data_cov = (true_cov * len(true_data) + false_cov * len(false_data)) / len(x)
    inv_cov = np.linalg.inv(data_cov)
    w0 = -0.5 * true_mean.dot(inv_cov).dot(true_mean) + 0.5 * false_mean.dot(inv_cov).dot(false_mean)
    w = inv_cov.dot(true_mean - false_mean)
    return w, w0, data_cov


def train_linear_logistic_regression(x, class_ids, wold, w0old):
    x = np.c_[x, np.ones(len(x))]
    wold = np.r_[wold, w0old]
    posteriors = logistic_sigmoid(x.dot(wold))
    r = posteriors * (1 - posteriors)
    wnew = wold - (posteriors - class_ids).dot(x).dot(inv((x.T * r).dot(x)))
    return wnew[:-1], wnew[-1]


def logistic_sigmoid(a):
    return 1 / (1 + np.exp(-a))


def mel_inv(x):
    return (np.exp(x/1127.)-1.)*700.


def mel(x):
    return 1127.*np.log(1. + x/700.)


def mel_filter_bank(nfft, nbands, fs, fstart=0, fend=None):
    if not fend:
      fend = 0.5 * fs
    cbin = np.round(mel_inv(np.linspace(mel(fstart), mel(fend), nbands + 2)) / fs * nfft).astype(int)
    mfb = np.zeros((nfft // 2 + 1, nbands))
    for ii in range(nbands):
        mfb[cbin[ii]:  cbin[ii+1]+1, ii] = np.linspace(0., 1., cbin[ii+1] - cbin[ii]   + 1)
        mfb[cbin[ii+1]:cbin[ii+2]+1, ii] = np.linspace(1., 0., cbin[ii+2] - cbin[ii+1] + 1)
    return mfb


def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift,a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def spectrogram(x, window, noverlap=None, nfft=None):
    if np.isscalar(window): window = np.hamming(window)
    if noverlap is None:    noverlap = window.size // 2
    if nfft     is None:    nfft     = window.size
    x = framing(x, window.size, window.size-noverlap)
    x = scipy.fftpack.fft(x*window, nfft)
    return x[:,:x.shape[1]//2+1]


def mfcc(s, window, noverlap, nfft, fs, nbanks, nceps):
    snrdb = 40
    noise = rand(s.shape[0])
    s = s + noise.dot(norm(s, 2)) / norm(noise, 2) / (10 ** (snrdb / 20))
    mfb = mel_filter_bank(nfft, nbanks, fs, 32)
    dct_mx = scipy.fftpack.idct(np.eye(nceps, nbanks), norm='ortho')
    S = spectrogram(s, window, noverlap, nfft)
    return dct_mx.dot(np.log(mfb.T.dot(np.abs(S.T)))).T


def wav16khz2mfcc(dir_name):
    features = {}
    for f in glob(dir_name + '/*.wav'):
        rate, s = wavfile.read(f)
        assert(rate == 16000)
        features[f] = mfcc(s, 400, 240, 512, 16000, 23, 13)
    return features


def png2fea(dir_name):
    features = {}
    for f in glob(dir_name + '/*.png'):
        features[f] = imread(f, True).astype(np.float64)
    return features
