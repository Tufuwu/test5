# Standard generating functions used in network science
#
# Copyright (C) 2021 Simon Dobson
#
# This file is part of epydemic, epidemic network simulations in Python.
#
# epydemic is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# epydemic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with epydemic. If not, see <http://www.gnu.org/licenses/gpl.html>.

from cmath import exp
from mpmath import polylog, zeta
from epydemic.gf import GF, gf_from_series, gf_from_coefficient_function


def gf_er(N: int, kmean: float = None, phi: float = None) -> GF:
    '''Return the generating function for the Poisson degree distribution of
    an ER network of N nodes with the given mean degree or occupation
    probability, as generated by :class:`epydemic.ERNetwork`.

    :param N: the number of nodes in the network
    :param kmean: (optional) the mean degree
    :param phi: (optional) the occupation probability
    :returns: the generating function'''

    # check we have exactly one of kmean or phi
    ps = len([p for p in [kmean, phi] if p is not None])
    if ps != 1:
        raise TypeError('Must provide either a mean degree or an occupation probability')

    # get the mean degree if it wasn't provided
    if kmean is None:
        kmean = N * phi

    # return the generating function
    return gf_from_series(lambda x: exp(kmean * (x - 1)))


def gf_powerlaw(exponent: float) -> GF:
    '''Return the generating function of the powerlaw
    degree distribution with the given exponent.

    :param exponent: the exponent of the distribution
    :returns: the generating function'''
    return gf_from_series(lambda x: polylog(exponent, x) / zeta(exponent))


def gf_ba(M: int) -> GF:
    '''Return the generating fuynction of the degree distribution of
    Barabasi-Albert network with connectivity of M edges per added
    node, as generated by class :class:`epydemic.BANetwork`. This is a
    powerlaw distribution with exponent 3.

    :param M: the number of edges per node added
    :returns: the generating function'''

    return gf_from_coefficient_function(lambda k: (2 * pow(M, 2)) / pow(k, 3))


def gf_plc(exponent: float, cutoff: float) -> GF:
    '''Return the generating function of the powerlaw-with-cutoff
    degree distribution given its exponent and cutoff, as generated
    by :class:`epydemic.PLCNetwork`.

    :param exponent: the exponent of the distribution
    :param cutoff: the cutoff
    :returns: the generating function'''
    return gf_from_series(lambda x: polylog(exponent, x * exp(-1 / cutoff)) / polylog(exponent, exp(-1 / cutoff)))
