from sortedcontainers import SortedDict

class SO3Tau(SortedDict):
    """
    Class for keeping track of multiplicity (number of channels) of a SO(3)
    vector. This is based upon an SortedDict class.
    """
    def __init__(self, data):
        if type(data)
