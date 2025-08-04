#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from pyomo.contrib.solver.common.config import PersistentBranchAndBoundConfig


class KNITROConfigMixin:
    pass


class KNITROConfig(PersistentBranchAndBoundConfig, KNITROConfigMixin):
    """
    Configuration class for KNITRO solver.
    This class can be extended to include specific configuration options for KNITRO.
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        PersistentBranchAndBoundConfig.__init__(
            self,
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        KNITROConfigMixin.__init__(self)
