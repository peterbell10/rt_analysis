
import mpmath as mp
import numpy as np

def sqrt(x):
    if isinstance(x, (int, float, np.ndarray)):
        return np.sqrt(x)
    else:
        return mp.sqrt(x)

def df1(x, rel_gamma, rel_e_He):
    """Proportional to the rate of change in f1"""
    return (1-x)**2 + (1-x) * rel_e_He - x * rel_gamma

def f1_quadratic(rel_gamma, rel_e_He):
    """Analytic solution for f1"""
    b = 2 + rel_e_He + rel_gamma
    c = 1 + rel_e_He
    return .5 * b * (1 - sqrt(1 - c * (2 / b)**2))


def df2(y, rel_gamma, rel_e_He):
    """Proportional to the rate of change in f2"""
    return y**2 + y * rel_e_He - (1-y) * rel_gamma

def f2_quadratic(rel_gamma, rel_e_He):
    """Analytic solution for f2"""
    b = rel_e_He + rel_gamma
    return .5  * (sqrt(b * b + 4 * rel_gamma) - b)


def f1_f2_iterative(f1_iv, rel_gamma, rel_e_He):
    """Iterative scheme from RT_CUDA"""
    f1 = f1_iv
    f2 = 1. - f1
    rel_alpha = 1 / rel_gamma
    delta = 1.
    tol = np.finfo(float).eps ** .5
    neutral = f1 > f2
    loop_count = 0
    while abs(delta) > tol and loop_count < 2000:
        rel_e = rel_e_He + f2
        if neutral:
            eold = f2
            f2 += f1 / (rel_e * rel_alpha)
            if (f2 > 1.):
                f2 = .9
                neutral = False

            f1 = 1. - f2
            var = f2
        else:
            eold = f1
            f1 += f2 * rel_e * rel_alpha
            f1 /= 2
            if f1 > 1.:
                f1 = .9
                neutral = True

            f2 = 1 - f1
            var = f1

        delta = var / eold - 1.
        loop_count += 1

    return (f1, f2, loop_count, delta)


def run_trial(rel_gamma, rel_e_He):
    """Compare the outputs of iterative and analytic scheme for given parameters"""
    args = (rel_gamma, rel_e_He)
    f1_analytic = f1_quadratic(*args)
    f2_analytic = f2_quadratic(*args)
    print('f1 and f2 from solving quadratic:')
    print('          f1:', f1_analytic)
    print('          f2:', f2_analytic)
    print('         df1: ', df1(f1_analytic, *args))
    print('         df2: ', df2(f2_analytic, *args))
    print('')

    f1_iter, f2_iter, loop_count, delta = f1_f2_iterative(.5, *args)
    print('f1 and f2 from iterative scheme:')
    print('          f1:', f1_iter)
    print('          f2:', f2_iter)
    print('  iterations:', loop_count)
    print('         tol:', delta)
    print('         df1:', df1(f1_iter, *args))
    print('         df2:', df2(f2_iter, *args))
    print('')

    ionization_dominant = args[0] > 1

    if ionization_dominant:
        abs_err = abs(f1_iter - f1_analytic)
        rel_err = abs_err / f1_iter
    else:
        abs_err = abs(f2_iter - f2_analytic)
        rel_err = abs_err / f2_iter

    print('Absolute error   = {:.3e}'.format(abs_err))
    print('Fractional error = {:.3e}'.format(rel_err))

