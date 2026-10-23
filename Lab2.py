#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI and Decision Making: Labs 1 and 2, as a standalone Python file.

Installation (once):
    python -m pip install numpy scipy matplotlib
Run (all tasks, reference checks, CSV, and plots):
    python ai_decision_making_lab1_lab2.py
Without plot windows / tests only:
    python ai_decision_making_lab1_lab2.py --no-show
    python ai_decision_making_lab1_lab2.py --self-test
Individual experiments (enter percentages on the command line):
    python ai_decision_making_lab1_lab2.py --method cap --cap 8 --k 3
    python ai_decision_making_lab1_lab2.py --method minvar --short
    python ai_decision_making_lab1_lab2.py --method weighted --alpha .5 --k 2
    python ai_decision_making_lab1_lab2.py --method floor --floor 6
    python ai_decision_making_lab1_lab2.py --method goal --p .45
    python ai_decision_making_lab1_lab2.py --method goal --return-goal 5 --tie-break
    python ai_decision_making_lab1_lab2.py --dataset notebook --method cap --cap 5 --k 6

Edit the data below or use --data file.json. JSON keys:
    names, mu, sigma, corr. mu and sigma must be DECIMAL VALUES, e.g. 0.08.
The internal Python functions also use decimal values exclusively.
Output: lab_results folder next to this file (can be changed with --out).

Scope: The PDF submission requires an HTML dashboard and LINGO models. This
file is the requested Python implementation of the mathematical tasks, including
plots, a comparison table, and explanations; it does not replace those file formats.
LINGO is not invoked; reference values come from the provided PDFs.

Corrections compared with the notebook:
- The official five assets are used by default; the notebook's ten assets remain selectable.
- All supports of sizes 1,...,K are considered; previously, all single-asset supports were missing.
- The risk limit may be inactive (e.g. a 20% cap -> 100% US equity, 16% risk).
- Equal returns, negative returns, and singular PSD matrices are handled.
- Explicit parameters replace global True/False switches and frozen defaults.
- Tolerant symmetry/eigenvalue checks; dimensions and finite values are validated.
- Negative weights are not clipped and then renormalized.
- A second stage is available for goal programming; normalization is fixed for sparse comparisons.

Methods: For positive-definite Sigma, Lab 1, the weighted sum, and the return
floor are solved analytically for every support. This is not a grid approximation.
For singular Sigma, convex SLSQP subproblems are used. Goal programming uses
scaled epigraph variables with analytical gradients (also not a grid search).
Numerical solutions are checked for solver status and feasibility.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, linprog


class InfeasibleError(ValueError):
    """Valid inputs, but no portfolio satisfies the hard constraints."""


class UnboundedError(ValueError):
    """Without short-sale limits, the return may be unbounded."""


@dataclass
class Data:
    names: list[str]
    mu: np.ndarray
    sigma: np.ndarray
    corr: np.ndarray

    def __post_init__(self):
        self.names = list(self.names)
        self.mu = np.array(self.mu, dtype=float, copy=True)
        self.sigma = np.array(self.sigma, dtype=float, copy=True)
        self.corr = np.array(self.corr, dtype=float, copy=True)
        self.n = len(self.names)
        if not 2 <= self.n <= 10:
            raise ValueError('Between 2 and 10 assets are allowed.')
        if self.mu.shape != (self.n,) or self.sigma.shape != (self.n,):
            raise ValueError('Names, returns, and volatilities must have matching lengths.')
        if self.corr.shape != (self.n, self.n):
            raise ValueError('The correlation matrix must be n x n.')
        if any(not str(x).strip() for x in self.names):
            raise ValueError('Asset names must not be empty.')
        if not all(np.isfinite(x).all() for x in (self.mu, self.sigma, self.corr)):
            raise ValueError('Inputs contain NaN or infinite values.')
        if np.any(self.sigma <= 0):
            raise ValueError('Volatilities must be positive.')
        if not np.allclose(self.corr, self.corr.T, atol=1e-10, rtol=0):
            raise ValueError('The correlation matrix is not symmetric.')
        if not np.allclose(np.diag(self.corr), 1, atol=1e-10, rtol=0):
            raise ValueError('The diagonal of the correlation matrix must be 1.')
        if np.max(np.abs(self.corr)) > 1 + 1e-10:
            raise ValueError('Correlations must lie between -1 and 1.')
        self.corr = (self.corr + self.corr.T) / 2
        self.eigenvalues = np.linalg.eigvalsh(self.corr)
        if self.eigenvalues[0] < -1e-10:
            raise ValueError('The correlation matrix is not positive semidefinite.')
        self.cov = self.sigma[:, None] * self.corr * self.sigma[None, :]
        # Numerical noise near zero does not imply a negative variance.
        if self.eigenvalues[0] < 0:
            e, q = np.linalg.eigh(self.cov)
            self.cov = (q * np.maximum(e, 0)) @ q.T


def official_data():
    return Data(['US equity', 'EU equity', 'Gov bonds', 'Corp bonds', 'Gold'],
                [.08, .07, .03, .045, .05], [.16, .18, .05, .08, .15],
                [[1, .8, -.2, .3, .05], [.8, 1, -.15, .3, .1],
                 [-.2, -.15, 1, .6, .25], [.3, .3, .6, 1, .15],
                 [.05, .1, .25, .15, 1]])


def notebook_data():
    """The original ten assets from notebook cell 0, unchanged."""
    
    Assets = [
        "US large cap",
        "US small cap",
        "EU equity",
        "Japan equity",
        "EM equity",
        "Gov bonds",
        "Corp IG",
        "High yield",
        "Gold",
        "REITs"
    ]
    
    n_assets = len(Assets)
    
    expected_returns = np.array([
        0.075, 0.090, 0.070, 0.060, 0.100,
        0.025, 0.035, 0.055, 0.045, 0.070
    ])
    
    sigma = np.array([
        0.15, 0.21, 0.17, 0.16, 0.23,
        0.05, 0.07, 0.11, 0.15, 0.19
    ])
    
    correlation_matrix = np.array([
        [1.00, 0.85, 0.80, 0.65, 0.70, -0.20, 0.25,  0.60, 0.05, 0.70],
        [0.85, 1.00, 0.75, 0.60, 0.70, -0.25, 0.20,  0.65, 0.05, 0.75],
        [0.80, 0.75, 1.00, 0.65, 0.70, -0.15, 0.25,  0.60, 0.10, 0.65],
        [0.65, 0.60, 0.65, 1.00, 0.60, -0.10, 0.20,  0.50, 0.10, 0.50],
        [0.70, 0.70, 0.70, 0.60, 1.00, -0.10, 0.30,  0.65, 0.20, 0.60],
        [-0.20,-0.25,-0.15,-0.10,-0.10, 1.00, 0.60, -0.05, 0.30, 0.05],
        [0.25, 0.20, 0.25, 0.20, 0.30,  0.60, 1.00,  0.55, 0.20, 0.35],
        [0.60, 0.65, 0.60, 0.50, 0.65, -0.05, 0.55,  1.00, 0.15, 0.60],
        [0.05, 0.05, 0.10, 0.10, 0.20,  0.30, 0.20,  0.15, 1.00, 0.10],
        [0.70, 0.75, 0.65, 0.50, 0.60,  0.05, 0.35,  0.60, 0.10, 1.00]
    ])
    
    return Data(Assets, expected_returns, sigma, correlation_matrix)


@dataclass
class Portfolio:
    weights: np.ndarray
    ret: float
    variance: float
    score: float = 0.0
    shortfall: float = 0.0
    variance_excess: float = 0.0
    stage1_score: float | None = None
    note: str = ''

    @property
    def vol(self):
        return float(np.sqrt(max(0., self.variance)))

    @property
    def holdings(self):
        return int(np.count_nonzero(np.abs(self.weights) > 1e-6))


class PortfolioLab:
    def __init__(self, data):
        self.data = data
        # Normalization always uses the unconstrained long-only five-/ten-asset case.
        self.mv = self.solve('minvar')
        self.mr = self.solve('maxret')
        self.dr = self.mr.ret - self.mv.ret
        self.dv = self.mr.variance - self.mv.variance

    def result(self, w):
        d = self.data
        return Portfolio(w.copy(), float(d.mu @ w), float(w @ d.cov @ w))

    def losses(self, w):
        q = self.result(w)
        # If there is no trade-off, solve() selects MV before division.
        return (q.variance - self.mv.variance) / self.dv, (self.mr.ret-q.ret) / self.dr

    @lru_cache(maxsize=32)
    def supports(self, k, long_only):
        d = self.data
        sizes = range(1, k+1) if long_only or k < d.n else [d.n]
        values = []
        for size in sizes:
            for ids in combinations(range(d.n), size):
                idx = np.array(ids)
                c = d.cov[np.ix_(idx, idx)]
                mu = d.mu[idx]
                try:
                    if np.linalg.eigvalsh(c)[0] <= 1e-12 * max(1., np.linalg.norm(c)):
                        raise np.linalg.LinAlgError
                    a = np.linalg.solve(c, np.ones(size))
                    b = np.linalg.solve(c, mu)
                    A, B = a.sum(), b.sum()
                    base, tilt = a/A, b-(B/A)*a
                    values.append((idx, c, mu, base, tilt, float(mu@tilt)))
                except np.linalg.LinAlgError:
                    values.append((idx, c, mu, None, None, None))
        return values

    def _numeric(self, c, mu, method, cap, floor, alpha, long_only):
        """Fallback for singular PSD matrices; no regularized substitute problem."""
        n = len(mu)
        scale = max(float(np.max(np.diag(c))), 1e-6)
        variance = lambda w: float(w @ c @ w)
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum()-1,
                        'jac': lambda w: np.ones(n)}]
        if method == 'cap':
            mv = self._numeric(c, mu, 'minvar', cap, floor, alpha, long_only)
            if variance(mv) > cap**2 + 1e-10:
                raise InfeasibleError('Cap is below the minimum volatility of this support.')
            if abs(variance(mv)-cap**2) < 1e-13:
                # With singular Sigma, returns may vary along the MV surface:
                # use a linear tie-break instead of an arbitrary MV portfolio.
                lp = linprog(-mu, A_eq=np.vstack([np.ones(n), c]),
                             b_eq=np.r_[1., c@mv],
                             bounds=[(0, 1)]*n if long_only else [(None,None)]*n,
                             method='highs')
                if lp.status == 3:
                    raise UnboundedError('Return is unbounded along a risk-free direction.')
                if not lp.success:
                    raise RuntimeError('Tie-break on the singular MV surface failed.')
                return lp.x
            constraints.append({'type': 'ineq', 'fun': lambda w: (cap**2-variance(w))/scale,
                                'jac': lambda w: -2*c@w/scale})
        if method == 'floor' and long_only and floor > mu.max()+1e-10:
            raise InfeasibleError('The return floor is unreachable on this support.')
        if method in ('floor', 'maxret'):
            target = floor if method == 'floor' else float(mu.max())
            constraints.append({'type': 'ineq', 'fun': lambda w: (mu@w-target)/.1,
                                'jac': lambda w: mu/.1})
        if method == 'cap':
            fun, jac = lambda w: -mu@w/.1, lambda w: -mu/.1
        elif method == 'weighted':
            fun = lambda w: alpha*variance(w)/self.dv-(1-alpha)*mu@w/self.dr
            jac = lambda w: 2*alpha*c@w/self.dv-(1-alpha)*mu/self.dr
        else:
            fun, jac = lambda w: variance(w)/scale, lambda w: 2*c@w/scale
        if not long_only and method in ('cap', 'weighted'):
            # A risk-free, budget-neutral return direction makes maximum return,
            # or any weighted sum with a return component, unbounded.
            lp = linprog(-mu, A_eq=np.vstack([np.ones(n), c]), b_eq=np.zeros(n+1),
                         bounds=[(-1, 1)]*n, method='highs')
            if lp.success and mu @ lp.x > 1e-9 and (method == 'cap' or alpha < 1):
                raise UnboundedError('Risk-free short arbitrage: the problem is unbounded.')
        opt = minimize(fun, np.full(n, 1/n), jac=jac, method='SLSQP',
                       bounds=[(0, 1)]*n if long_only else None,
                       constraints=constraints, options={'ftol': 1e-12, 'maxiter': 500})
        if not opt.success:
            # Never silently report a solver failure as proven infeasibility.
            raise RuntimeError(f'Singular subproblem: {opt.message}')
        return opt.x

    def solve(self, method, *, cap=.08, floor=.06, alpha=.5, k=None, long_only=True):
        """minvar / maxret (without cap, long-only) / cap / floor / weighted.

        In short-selling models only sum(w)=1 remains; there is no hidden leverage limit.
        K counts all nonzero weights, including negative positions.
        """
        d = self.data
        if method not in {'minvar', 'maxret', 'cap', 'floor', 'weighted'}:
            raise ValueError('Unknown method.')
        k = d.n if k is None else k
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= d.n:
            raise ValueError('K must be an integer between 1 and n.')
        if not all(np.isfinite(x) for x in (cap, floor, alpha)) or cap < 0:
            raise ValueError('Parameters must be finite; cap must not be negative.')
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be between 0 and 1.')
        if method == 'maxret' and not long_only:
            if np.ptp(d.mu) > 1e-12 and k > 1:
                raise UnboundedError('Maximum return without a cap is unbounded with short selling.')
        if method == 'weighted':
            if self.dr <= 1e-12 or self.dv <= 1e-14:
                return self.solve('minvar', k=k, long_only=long_only)
            if alpha == 0:
                return self.solve('maxret', k=k, long_only=long_only)
        if method == 'floor' and long_only and floor > max(d.mu)+1e-10:
            raise InfeasibleError(f'Return floor {floor:.2%} > maximum return {max(d.mu):.2%}.')
        best, best_value = None, np.inf
        for idx, c, mu, base, tilt, H in self.supports(k, long_only):
            if len(idx) == 1:
                w = np.ones(1)
            elif method == 'maxret' and long_only:
                # In a return tie, minimize variance on the maximum-return surface.
                if np.ptp(mu) > 1e-12:
                    continue
                w = base if base is not None else self._numeric(c, mu, 'minvar', cap, floor, alpha, True)
            elif base is None:
                try:
                    w = self._numeric(c, mu, method, cap, floor, alpha, long_only)
                except InfeasibleError:
                    continue
            elif method in ('minvar', 'maxret'):
                w = base
            elif method == 'weighted':
                w = base + (1-alpha)*self.dv/(2*alpha*self.dr)*tilt
            elif method == 'floor':
                if mu @ base >= floor-1e-12:
                    w = base
                elif H > 1e-15:
                    w = base + (floor-mu@base)/H*tilt
                else:
                    continue
            else:  # cap: formula for the upper efficient half.
                vmin = float(base @ c @ base)
                if cap**2 < vmin-1e-12:
                    continue
                if H <= 1e-15:
                    w = base  # Constant return: use MV as the tie-break.
                else:
                    w = base + np.sqrt(max(0., cap**2-vmin)/H)*tilt
            if not np.isfinite(w).all() or abs(w.sum()-1) > 1e-7:
                continue
            if long_only and w.min() < -1e-9:
                continue
            full = np.zeros(d.n)
            full[idx] = w
            q = self.result(full)
            if method == 'cap' and q.variance > cap**2 + 1e-10:
                continue
            if method == 'floor' and q.ret < floor-1e-9:
                continue
            if method in ('minvar', 'floor'):
                value = q.variance
            elif method in ('maxret', 'cap'):
                value = -q.ret
            else:
                lv, lr = self.losses(full)
                value = alpha*lv + (1-alpha)*lr
            if (value < best_value-1e-13 or
                (abs(value-best_value) <= 1e-13 and best is not None and q.variance < best.variance)):
                best, best_value = q, value
        if best is None:
            raise InfeasibleError(f'No feasible portfolio: {method}, K={k}, long_only={long_only}.')
        best.score = best_value if method == 'weighted' else 0.
        if method == 'cap':
            best.note = 'cap binds' if abs(best.vol-cap) < 1e-7 else 'cap slack'
        if method == 'floor':
            best.note = 'floor binds' if abs(best.ret-floor) < 1e-7 else 'floor slack'
        return best

    def goal(self, *, return_goal=.06, volatility_goal=.08, p=.45,
             tie_break=False, tau=1e-8):
        """Convex goal programming, long-only, without a cardinality limit.

        x=(w, ur, uv), where ur=dr/Delta_r and uv=dv/Delta_v.
        Stage 2 minimizes F_0.5 subject to G <= G* + tau.
        """
        if not all(np.isfinite(x) for x in (return_goal, volatility_goal, p, tau)):
            raise ValueError('Goal parameters must be finite.')
        if not 0 < p < 1 or volatility_goal < 0 or tau < 0:
            raise ValueError('Required: 0<p<1, volatility_goal>=0, and tau>=0.')
        if self.dr <= 1e-12 or self.dv <= 1e-14:
            raise ValueError('Degenerate trade-off: goal normalization is undefined.')
        d, n = self.data, self.data.n
        grad_score = np.r_[np.zeros(n), p, 1-p]
        def score(x):
            return float(grad_score @ x)
        def residuals(x):
            w, ur, uv = x[:n], x[n], x[n+1]
            return np.array([(d.mu@w-return_goal)/self.dr+ur,
                             (volatility_goal**2-w@d.cov@w)/self.dv+uv])
        def jac_residuals(x):
            return np.vstack([np.r_[d.mu/self.dr, 1., 0.],
                              np.r_[-2*d.cov@x[:n]/self.dv, 0., 1.]])
        constraints = [
            {'type': 'eq', 'fun': lambda x: x[:n].sum()-1,
             'jac': lambda x: np.r_[np.ones(n), 0., 0.]},
            {'type': 'ineq', 'fun': residuals, 'jac': jac_residuals}]
        w0 = self.mv.weights
        x0 = np.r_[w0, max(0., (return_goal-self.mv.ret)/self.dr),
                   max(0., (self.mv.variance-volatility_goal**2)/self.dv)]
        def optimize(fun, jac, start):
            opt = minimize(fun, start, jac=jac, method='SLSQP',
                           bounds=[(0., 1.)]*n+[(0., None)]*2,
                           constraints=constraints,
                           options={'ftol': 1e-12, 'maxiter': 500})
            if (not opt.success or abs(opt.x[:n].sum()-1) > 1e-8 or
                    min(residuals(opt.x)) < -1e-8):
                raise RuntimeError(f'Goal optimization failed: {opt.message}')
            return opt.x
        x = optimize(score, lambda x: grad_score, x0)
        gstar = score(x)
        if tie_break:
            constraints.append({'type': 'ineq', 'fun': lambda z: gstar+tau-score(z),
                                'jac': lambda z: -grad_score})
            def f(z):
                lv, lr = self.losses(z[:n])
                return .5*(lv+lr)
            def df(z):
                return np.r_[d.cov@z[:n]/self.dv-.5*d.mu/self.dr, 0., 0.]
            x = optimize(f, df, x)
            if score(x) > gstar+tau+1e-8:
                raise RuntimeError('Goal tie-break violates the stage-1 bound.')
        q = self.result(x[:n])
        q.shortfall = max(return_goal-q.ret, 0.)
        q.variance_excess = max(q.variance-volatility_goal**2, 0.)
        q.score = p*q.shortfall/self.dr+(1-p)*q.variance_excess/self.dv
        q.stage1_score = gstar
        missed = []
        if q.shortfall > 1e-7:
            missed.append('return')
        if q.variance_excess > 1e-7:
            missed.append('variance')
        q.note = 'misses '+', '.join(missed) if missed else 'goals met (numerical tolerance)'
        return q


DISCUSSION = '''LAB 2: EXPLANATIONS
1. US equity alone dominates EU equity alone: it has a higher return (8% vs. 7%)
   and lower risk (16% vs. 18%). However, EU equity can still be useful in
   diversified portfolios: covariances with the overall portfolio are decisive.
   The EU allocation in the MV portfolio is approximately 0.908%.
2. The goals are combined linearly. The variance w.T @ Sigma @ w remains
   quadratic; for alpha>0, the weighted sum is a convex quadratic program.
3. alpha weights risk, while p weights the return shortfall. Normalization is
   calculated from the two anchor portfolios and held fixed for comparisons across K.
4. Both alpha=0 and alpha=.25 choose 100% US equity: a range of preferences
   can select the same corner point of the feasible simplex.
5. A 6% return requires at least 8.7659467% volatility. Under an 8% cap,
   the maximum possible return is 5.7126685%. The two hard limits are therefore
   infeasible together. A 9% floor and a 3% cap are also infeasible.
6. Goal programming allows positive deviations. At p=.25 only the return goal
   is missed; at p=.45 both goals are missed; at p=.50 only the variance goal
   is missed. For p=.45, the optimum lies between the cap and floor solutions:
   their marginal costs are equal. The frontier slope dr/dv there is
   (1-p)*Delta_r/(p*Delta_v).
7. For compatible goals (5%, 8%), G=0 for many portfolios, including
   dominated portfolios. Stage 2 minimizes F_.5 subject to G<=G*+1e-8. The
   tolerance permits a very small deviation from the exact lexicographic optimum.
8. "I MUST stay below 8%": use a hard cap. "I WANT a 6% return and about 8%
   risk": use soft goals with an agreed value of p. Expected return is not guaranteed.
9. K=3 does not change F_.5 because the unconstrained optimum holds three assets.
   K=2 restricts the choice; K=1 permits only single-asset portfolios. A smaller
   K cannot improve the optimal frontier. The union of supports is non-convex,
   so the sparse frontier need not be concave, and some efficient points can
   only be reached using epsilon constraints.
10. The variance deviation dv has squared decimal units; return shortfalls are
    reported in percentage points. G-scores with different values of p are not
    directly comparable.
'''


def experiments(lab):
    """Run all Lab 1 and Lab 2 cases; report infeasibility explicitly, not stale values."""
    rows = []
    def run(label, method=None, **kw):
        try:
            q = lab.goal(**kw) if method == 'goal' else lab.solve(method, **kw)
            rows.append((label, q, 'optimal'))
        except InfeasibleError as e:
            rows.append((label, None, 'infeasible: '+str(e)))
    for k, lo in [(5, True), (5, False), (2, True)]:
        run(f'Lab1 MV K={k} long={lo}', 'minvar', k=k, long_only=lo)
    for k, lo, cap in [(5, True, .08), (3, True, .08), (2, True, .08),
                       (5, False, .08), (5, True, .12), (5, True, .03)]:
        run(f'Lab1 cap={cap:g} K={k} long={lo}', 'cap', cap=cap, k=k, long_only=lo)
    for a in [0, .25, .5, .75, 1]:
        run(f'alpha={a:g}', 'weighted', alpha=a)
    for r in [.04, .05, .06, .07, .08, .09]:
        run(f'floor={r:g}', 'floor', floor=r)
    for s in [.03, .06, .08, .10, .12, .20]:
        run(f'cap={s:g}', 'cap', cap=s)
    for p in [.25, .45, .5]:
        run(f'GP p={p:g}', 'goal', p=p)
    run('GP compatible stage1', 'goal', p=.5, return_goal=.05)
    run('GP compatible stage2', 'goal', p=.5, return_goal=.05, tie_break=True)
    for k in [5, 3, 2, 1]:
        run(f'Sparse K={k}', 'weighted', alpha=.5, k=k)
    # Both directions use unrounded values, not approximations from the table.
    a = lab.solve('floor', floor=.06)
    b = lab.solve('cap', cap=.08)
    run('inverse cap from floor6', 'cap', cap=a.vol)
    run('inverse floor from cap8', 'floor', floor=b.ret)
    return rows


# Reference values in percent from the PDFs; not used for optimization.
# Lab 1: tolerance .1 percentage points. Lab 2: .05 percentage points.
REFERENCES = [
 ('Lab1 MV K=5 long=True', [12.2,.9,86.1,0,.8],3.66,4.43,.1),
 ('Lab1 MV K=5 long=False',[14.7,2.3,103.5,-20.6,.1],3.52,4.28,.1),
 ('Lab1 MV K=2 long=True',[13.1,0,86.9,0,0],3.65,4.43,.1),
 ('Lab1 cap=0.08 K=5 long=True',[40.5,0,19.1,24.3,16.1],5.71,8,.1),
 ('Lab1 cap=0.08 K=3 long=True',[46.8,0,36.9,0,16.3],5.67,8,.1),
 ('Lab1 cap=0.08 K=2 long=True',[50.7,0,49.3,0,0],5.54,8,.1),
 ('Lab1 cap=0.08 K=5 long=False',[50.6,-12.5,19,26.2,16.7],5.76,8,.1),
 ('Lab1 cap=0.12 K=5 long=True',[69.7,0,0,9.2,21.1],7.04,12,.1),
 ('alpha=0',[100,0,0,0,0],8,16,.05),
 ('alpha=0.25',[100,0,0,0,0],8,16,.05),
 ('alpha=0.5',[48.133,0,0,32.030,19.837],6.2838,9.5472,.05),
 ('alpha=0.75',[26.821,0,66.014,0,7.165],4.4843,5.1995,.05),
 ('alpha=1',[12.187,.908,86.127,0,.777],3.6612,4.4277,.05),
 ('floor=0.04',[18.633,0,77.949,0,3.419],4,4.5656,.05),
 ('floor=0.05',[32.695,0,46.599,9.776,10.930],5,6.2460,.05),
 ('floor=0.06',[43.685,0,7.997,30.119,18.199],6,8.7659,.05),
 ('floor=0.07',[68.419,0,0,10.516,21.065],7,11.8454,.05),
 ('floor=0.08',[100,0,0,0,0],8,16,.05),
 ('cap=0.06',[31.475,0,50.885,7.517,10.123],4.8890,6,.05),
 ('cap=0.08',[40.527,0,19.089,24.274,16.111],5.7127,8,.05),
 ('cap=0.1',[52.491,0,0,27.408,20.101],6.4377,10,.05),
 ('cap=0.12',[69.668,0,0,9.192,21.140],7.0441,12,.05),
 ('cap=0.2',[100,0,0,0,0],8,16,.05),
 ('GP p=0.25',[40.527,0,19.089,24.274,16.111],5.7127,8,.05),
 ('GP p=0.45',[42.040,0,13.774,27.075,17.111],5.8504,8.3639,.05),
 ('GP p=0.5',[43.685,0,7.997,30.119,18.199],6,8.7659,.05),
 ('GP compatible stage2',[40.527,0,19.089,24.274,16.111],5.7127,8,.05),
 ('Sparse K=5',[48.133,0,0,32.030,19.837],6.2838,9.5472,.05),
 ('Sparse K=3',[48.133,0,0,32.030,19.837],6.2838,9.5472,.05),
 ('Sparse K=2',[49.732,0,0,50.268,0],6.2406,9.9341,.05),
 ('Sparse K=1',[0,0,0,100,0],4.5,8,.05),
]


def self_test(lab=None, rows=None):
    lab = PortfolioLab(official_data()) if lab is None else lab
    rows = experiments(lab) if rows is None else rows
    got = {name: q for name,q,_ in rows}
    def check(ok, message):
        if not ok:
            raise AssertionError(message)
    for name, weights, ret, vol, tol in REFERENCES:
        q = got[name]
        check(q is not None, name+' is missing')
        actual = 100*np.r_[q.weights,q.ret,q.vol]
        expected = np.r_[weights,ret,vol]
        check(np.max(np.abs(actual-expected)) <= tol, name+f': {actual} != {expected}')
    for name in ['floor=0.09', 'cap=0.03', 'Lab1 cap=0.03 K=5 long=True']:
        check(got[name] is None, name+' must be infeasible')
    for name,q,_ in rows:
        if q is None:
            continue
        check(abs(q.weights.sum()-1)<1e-6, name+': budget')
        if 'long=False' not in name:
            check(q.weights.min() >= -1e-6, name+': long-only')
        if name.startswith('cap='):
            check(q.vol <= float(name.split('=')[1])+1e-6, name+': cap')
        if name.startswith('floor='):
            check(q.ret >= float(name.split('=')[1])-1e-6, name+': floor')
    for left,right in [('inverse cap from floor6','floor=0.06'),
                       ('inverse floor from cap8','cap=0.08')]:
        check(np.max(np.abs(got[left].weights-got[right].weights))<1e-6, 'Inverse: '+left)
    check(abs(got['GP p=0.45'].score-.0293761)<1e-6, 'Goal score')
    check(abs(got['alpha=0.5'].score-.349093)<1e-6, 'Weighted score')
    check(got['GP compatible stage2'].score <= got['GP compatible stage2'].stage1_score+1.1e-8,
          'Lexicographic bound')
    # Regression tests for actual notebook bugs.
    equal = PortfolioLab(Data(['A','B'], [.03,.03], [.1,.2], np.eye(2)))
    check(abs(equal.solve('cap', cap=.5).ret-.03)<1e-10, 'Equal returns')
    single = lab.solve('cap', cap=.20, k=1)
    check(single.weights[0] == 1, 'Single asset / inactive risk limit')
    neg = PortfolioLab(Data(['A','B'], [-.03,-.04], [.1,.2], np.eye(2)))
    check(neg.solve('cap', cap=.5).ret < 0, 'Negative returns')
    singular = PortfolioLab(Data(['A','B'], [.04,.04], [.1,.1], np.ones((2,2))))
    check(abs(singular.solve('cap', cap=.2).vol-.1)<1e-7, 'Singular PSD matrix')
    try:
        singular.solve('cap', cap=.05)
    except InfeasibleError:
        pass
    else:
        raise AssertionError('Singular matrix: infeasible cap accepted')
    try:
        Data(['A','B','C'], [.1]*3,[.1]*3,[[1,.9,-.9],[.9,1,.9],[-.9,.9,1]])
    except ValueError:
        pass
    else:
        raise AssertionError('Non-PSD matrix was accepted')
    print(f'PASS: {len(REFERENCES)} reference portfolios, 3 infeasible cases, '
          'hard constraints, inverse epsilon models, scores, and edge cases.')


def report(rows, data, out):
    out.mkdir(parents=True, exist_ok=True)
    with (out/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['case','status']+[x+' weight (%)' for x in data.names]+
                        ['return (%)','volatility (%)','variance','holdings','score',
                         'return shortfall (pp)','variance excess','stage1 score','remark'])
        for name,q,status in rows:
            if q is None:
                writer.writerow([name,status]+['']*(data.n+10))
                print(f'{name:<37} INFEASIBLE')
            else:
                writer.writerow([name,status,*list(q.weights*100),q.ret*100,q.vol*100,
                                 q.variance,q.holdings,q.score,q.shortfall*100,
                                 q.variance_excess,q.stage1_score,q.note])
                print(f'{name:<37} r={q.ret:8.4%}  s={q.vol:8.4%}  '
                      f'score={q.score:.7f}  {q.note}')
                print('    weights (%): '+', '.join(f'{100*w:.4f}' for w in q.weights))
    (out/'discussion.txt').write_text(DISCUSSION, encoding='utf-8')


def plots(lab, rows, out, show=True, k=None, long_only=True):
    import matplotlib.pyplot as plt
    d = lab.data
    mv = lab.solve('minvar', k=k, long_only=long_only)
    # Maximum plotted cap is a display choice, not a bound in a Short model.
    end = max(d.sigma)*1.25 if not long_only else max(d.sigma)
    grid = np.linspace(mv.vol, end, 65)
    points = [lab.solve('cap', cap=s, k=k, long_only=long_only) for s in grid]
    fig, ax = plt.subplots(figsize=(9,6), layout='constrained')
    ax.plot([q.vol*100 for q in points], [q.ret*100 for q in points], label='Efficient frontier')
    ax.scatter(d.sigma*100,d.mu*100,color='gray',label='Assets')
    for i,name in enumerate(d.names):
        ax.annotate(name,(d.sigma[i]*100,d.mu[i]*100),xytext=(4,5),textcoords='offset points',fontsize=8)
    choices = {'alpha=0.5','cap=0.08','floor=0.06','GP p=0.45', 'GP compatible stage2'}
    selected = [(name,q) for name,q,_ in rows if q is not None and
                (name in choices or len(rows)==1)]
    for name,q in selected:
        ax.scatter(q.vol*100,q.ret*100,s=75,label=name,zorder=4)
    if len(rows)>1:
        ax.scatter(8,6,marker='x',color='red',s=90,label='Desired pair (8%, 6%)')
    ax.set(xlabel='Volatility (%)',ylabel='Expected return (%)',title='Risk-return plane')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(out/'frontier.png',dpi=170)
    if selected:
        fig2, axs = plt.subplots(len(selected),1,figsize=(9,3*len(selected)),
                                 layout='constrained',squeeze=False)
        for axis,(name,q) in zip(axs[:,0],selected):
            axis.bar(d.names,100*q.weights,color='#238b8d')
            axis.axhline(0,color='gray',linewidth=.7)
            axis.set(title=f'{name}: r={q.ret:.4%}, s={q.vol:.4%}',ylabel='Weight (%)')
            axis.grid(axis='y',alpha=.2)
            axis.tick_params(axis='x',labelsize=8)
        fig2.savefig(out/'weights.png',dpi=170)
    if show:
        plt.show()
    else:
        plt.close('all')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset',choices=['official','notebook'],default='official')
    parser.add_argument('--data',type=Path,help='Custom data as JSON (decimal values)')
    parser.add_argument('--method',choices=['all','minvar','maxret','cap','floor','weighted','goal'],default='all')
    parser.add_argument('--cap',type=float,default=8,help='Volatility cap in percent')
    parser.add_argument('--floor',type=float,default=6,help='Return floor in percent')
    parser.add_argument('--alpha',type=float,default=.5)
    parser.add_argument('--p',type=float,default=.45)
    parser.add_argument('--return-goal',type=float,default=6,help='Return goal in percent')
    parser.add_argument('--volatility-goal',type=float,default=8,help='Volatility goal in percent')
    parser.add_argument('--tie-break',action='store_true')
    parser.add_argument('--k',type=int,help='Maximum number of assets (default: all)')
    parser.add_argument('--short',action='store_true',help='Short selling (Lab 1, floor, and weighted only)')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'lab_results')
    parser.add_argument('--no-show',action='store_true',help='Save plots without opening a window')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    data = Data(**json.loads(args.data.read_text(encoding='utf-8'))) if args.data else (
        official_data() if args.dataset=='official' else notebook_data())
    lab = PortfolioLab(data)
    print('Eigenvalues R:', np.round(data.eigenvalues,8))
    print(f'Anchors: MV r={lab.mv.ret:.6%}, s={lab.mv.vol:.6%}; '
          f'MR r={lab.mr.ret:.6%}, s={lab.mr.vol:.6%}')
    print(f'Fixed scales: Delta_r={lab.dr:.15f}, Delta_v={lab.dv:.15f}')
    if args.method == 'all':
        if args.dataset != 'official' or args.data or args.k is not None or args.short:
            parser.error('method=all is the official comparison series; choose a method for custom data/K/short selling.')
        rows = experiments(lab)
        self_test(lab,rows)
    elif args.method == 'goal':
        if args.short or args.k is not None:
            parser.error('Goal programming: core model uses long-only positions and all assets.')
        q = lab.goal(return_goal=args.return_goal/100,volatility_goal=args.volatility_goal/100,
                     p=args.p,tie_break=args.tie_break)
        rows = [('Selected goal',q,'optimal')]
    else:
        q = lab.solve(args.method,cap=args.cap/100,floor=args.floor/100,alpha=args.alpha,
                      k=args.k,long_only=not args.short)
        rows = [('Selected '+args.method,q,'optimal')]
    report(rows,data,args.out)
    plots(lab,rows,args.out,show=not args.no_show,k=args.k,long_only=not args.short)
    print('\n'+DISCUSSION if args.method=='all' else '')
    print(f'Results saved: {args.out.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f'ERROR: {error}') from error
