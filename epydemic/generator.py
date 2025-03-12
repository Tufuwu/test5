# Network generator base class and basic instances
#
# Copyright (C) 2017--2020 Simon Dobson
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

import sys
from networkx import Graph
from typing import Dict, Any, Optional, Iterator
if sys.version_info >= (3, 8):
    from typing import Final
else:
    # backport compatibility with older typing
    from typing_extensions import Final


class NetworkGenerator(object):
    '''An object that generates instances of networks on demand.

    In stantsitical physics, an *ensemble* is a family of structures
    generated by a stochastic process with some specified parameters
    :cite:`StatPhy`.  In network science we often deal with ensembles
    of networks, for example the ensemble of all networks constructed
    by independently removing edges with some fixed probability (the
    ER networks), and are looking to find the "ensemble average" of
    some property, the value it takes on a "representative" network.

    In ``epydemic``, a network generator represents an ensemble of networks
    that can be sampled to acquire random instances needed for experiments.
    the parameters of the ensemble are taken from the experimental parameters.
    Sub-classes override the :meth:`_generate` method to create new instances
    from the ensemble.

    A generator is driven by experimental parameters passed to the
    experiment represented by the :class:`NetworkExperiment` class. These can be
    set at construction and/or re-set using the :meth:`set` method at
    any time.

    The maximum number of instances the generator can generate can also be limited,
    which essentially creates a set of networks of the given class that are
    generated on demand. The limit can't be changed and applies to the total number
    of networks generated, both singly (using :meth:`generate`) and by
    iteration (using :meth:`__iter__`).

    :param params: (optional) experimental parameters
    :param limit: (optional) maximum number of network instances to generate (defaults to unbounded)

    '''

    # Extra parameter added automatically
    TOPOLOGY: Final[str] = 'topology'   #: Experimental parameter flagging the network's topology.

    def __init__(self, params: Dict[str, Any] = None, limit: Optional[int] = None):
        if params is None:
            params = dict()
        self._params: Dict[str, Any] = params
        self._remaining: Optional[int] = limit


    # ---------- Network generation ----------

    def set(self, params: Dict[str, Any]) -> 'NetworkGenerator':
        '''Set the parameters for the family of networks generated by this generator.

        :param params: the parameters
        :returns: the generator'''
        self._params = params.copy()
        return self

    def topology(self) -> str:
        '''Return the topoology flag for this generator. This should be overridden by
        sub-classes.

        :returns: the topology'''
        raise NotImplementedError('NetworkGenerator.topology needs to be overridden by sub-classes')

    def _generate(self, params: Dict[str, Any]) -> Graph:
        '''Generate an instance of a network. This method isn't called directly,
        but can be accessed through either the :meth:`generate` method or through
        the iterator interface, Sub-classes should override this method to provide
        the code needed to create instances of the network family described by the
        generator, according to the experimental paramaters.

        :param params: the experimental parameters
        :returns: a network instance'''
        raise NotImplementedError('NetworkGenerator._generate needs to be overridden by sub-classes')

    def generate(self) -> Optional[Graph]:
        '''Generate a new instance of a network. The method returns
        None if the total number of allowed instances have already been generated.

        :returns: a network instance or None'''
        if self._remaining is None:
            return self._generate(self._params)
        elif self._remaining > 0:
            self._remaining -= 1
            return self._generate(self._params)
        else:
            return None


    # ---------- Iterator interface ----------

    def __iter__(self) -> Iterator[Graph]:
        '''Return iterator over the family of networks this generator generates.

        :returns: the generator itself'''
        return self

    def __next__(self) -> Graph:
        '''Generate an instance of the network as long as the maximum number
        of instances (if any) have not already been generated.

        :returns: a network'''
        g = self.generate()
        if g is None:
            raise StopIteration()
        else:
            return g
