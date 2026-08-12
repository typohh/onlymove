"""Axis warm start derived INSIDE the joint fit, from the model's own residuals.

The production warm start (`utils.rating.residual_axes.eigen_directions`) needs
an axis-free 4PL to take residuals against, and in the two-stage pipeline one
exists: `build_axes` runs after `train_rating` and reads the frozen fit. A
single-stage model has no such object at initialization, so the joint trainer
borrowed the SHIPPED `ratings/` — which makes the "one model, one pass" claim
false at init, and, worse, leaks: the shipped fit saw every user, so a
`--holdout-user-frac` run warm-started from directions informed by the very
accounts being scored (measured at 17.2% / 13.6% of the warm-start users).

The model supplies the missing object itself. At v = 0 the loading gradient is
identically zero — which is the whole reason a warm start is needed — so a
burn-in with v pinned at zero trains EXACTLY the axis-free baseline the residual
scores are defined against. It is not a separate stage and not wasted work:
those epochs are the rating heads' own fit, which continues afterwards.

    s_u = sum_p (y_up - p_hat_up) * x_p / n_u

with p_hat from the model at v = 0, over TRAINING ROWS ONLY — so held-out users
and held-out rows are excluded by construction rather than by a filter someone
has to remember to apply.

TWO PROPERTIES ARE LOAD-BEARING and are carried over from the production form:

  CENTRING across users. The uncentred lead direction is the one in which the
  baseline mispredicts for EVERYONE, i.e. population residual difficulty — a
  property of problems, not of people. Seeding from it puts difficulty into
  every axis at initialization for the mean penalty to fight back out.

  DIVISION BY n_u. s_u is a SUM over a user's attempts, so |s_u| grows with how
  much they played and its weight in a second moment grows with n_u SQUARED. On
  the raw form a single heavy account supplied 24-36% of every axis. Dividing by
  n_u is also the only normalisation for which centring removes population
  difficulty exactly, since that term enters s_u proportionally to n_u.

ONE PROPERTY IS DELIBERATELY DROPPED. `_user_score_matrix` regresses out a
per-user [1, rank] design before taking residuals, because `train_axes` carries
an explicit per-user rank slope `beta_u` and seeding axes with what beta_u will
absorb is wasteful. THE JOINT MODEL HAS NO SUCH PARAMETER — its logit is
a_p*(theta_u - b_p) + sum_k g_k*v_k, where theta is a per-profile level and the
slope is per PROBLEM. A user's rank-sensitivity is therefore something the axes
can and should represent, so projecting it out here would delete signal rather
than avoid duplicating it.
"""

import numpy as np


def burn_in_directions(m, tfeat, tp, tprof, tu, ty, tr, k, n_users,
                       batch=65536):
    """Top-k escape directions from the model's OWN axis-free residuals.

    Call after the burn-in epochs, with the model's v still at zero. Returns
    (k, fdim) float32 with unit rows, ready for m.g_lin.weight.

    Requires v == 0: the residual is defined under the axis-free baseline, and
    a non-zero v would make p_hat depend on the very loadings being chosen."""
    torch = m.torch
    fdim = tfeat.shape[1]
    S = torch.zeros(n_users, fdim, dtype=torch.float64, device=tfeat.device)
    n = torch.zeros(n_users, dtype=torch.float64, device=tfeat.device)

    was_training = m.training
    m.train(False)                      # dropout off: the baseline, not a sample
    with torch.no_grad():
        assert float(m.v.abs().max()) == 0.0, (
            "burn_in_directions requires v == 0; the residual is defined under "
            "the axis-free baseline")
        for i in range(0, len(tr), batch):
            bi = tr[i:i + batch]
            pred, _g = m.forward(tprof[bi], tu[bi], tfeat[tp[bi]])
            resid = (ty[bi] - pred).double()
            x = tfeat[tp[bi]].double()
            S.index_add_(0, tu[bi], resid[:, None] * x)
            n.index_add_(0, tu[bi], torch.ones_like(resid))
    m.train(was_training)

    S = S.cpu().numpy()
    n = n.cpu().numpy()
    seen = n > 0                        # users with no TRAINING rows contribute
    S = S[seen] / n[seen, None]         # nothing — held-out users land here
    S = S - S.mean(axis=0, keepdims=True)
    _ev, vec = np.linalg.eigh(S.T @ S / S.shape[0])
    top = vec[:, ::-1][:, :k].T
    return (top / (np.linalg.norm(top, axis=1, keepdims=True) + 1e-9)
            ).astype(np.float32), int(seen.sum())
