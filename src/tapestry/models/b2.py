"""B2: B1's direct forecaster with separately masked historical covariates."""

from .b1 import B1


class B2(B1):
    """The B1 direct backbone plus covariates in its shared context encoder."""

    def __init__(self, *args, **kwargs):
        if kwargs.get('direct') is False:
            raise ValueError('B2 supports the direct B1 pipeline only')
        kwargs['direct'] = True
        super().__init__(*args, **kwargs)
