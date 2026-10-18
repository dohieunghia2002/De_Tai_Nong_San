"""M4 - harvest decision per fruit (ready / not ready / defer) and decision-level metrics."""
import numpy as np

READY, NOT_READY, DEFER = 'ready', 'not_ready', 'defer'


def decide(p, half_width, T):
    """ready if the interval [p - q, p + q] lies at or above T, not ready if it lies below T, otherwise defer
    (destructive test or wait). half_width = 0 gives the point decision without deferral."""
    p = np.asarray(p, dtype=float)
    decided = np.abs(p - T) >= half_width
    return np.where(~decided, DEFER, np.where(p >= T, READY, NOT_READY))


def decision_metrics(y, p, half_width, T):
    """Shares of all fruit (*_no_defer, interval_coverage, decided) or of the decided fruit (*_decided).
    false_ready: declared ready while the true DMC is below T, i.e. immature fruit harvested.
    missed_ready: declared not ready while the true DMC is at or above T."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    truly_ready = y >= T
    said_ready = p >= T
    correct = said_ready == truly_ready
    false_ready = said_ready & ~truly_ready
    missed_ready = ~said_ready & truly_ready
    decided = decide(p, half_width, T) != DEFER

    def share(mask, among=None):
        if among is None:
            return float(mask.mean())
        return float(mask[among].mean()) if among.any() else float('nan')

    return {
        'n_fruit': int(y.size),
        'truly_ready': share(truly_ready),
        'accuracy_no_defer': share(correct),
        'false_ready_no_defer': share(false_ready),
        'missed_ready_no_defer': share(missed_ready),
        'interval_coverage': share(np.abs(p - y) <= half_width),
        'decided': share(decided),
        'accuracy_decided': share(correct, decided),
        'false_ready_decided': share(false_ready, decided),
        'missed_ready_decided': share(missed_ready, decided),
    }
