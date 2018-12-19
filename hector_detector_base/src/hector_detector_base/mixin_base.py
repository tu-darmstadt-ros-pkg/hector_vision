class MixinBase(object):
    """
    This is a base class for all classes here that may use mixins to prevent the issue where you have to call object's
    __init__ which doesn't take parameters but then the next __init__ in line won't get its arguments.
    """
    def __init__(self, *args, **kwargs):
        super(MixinBase, self).__init__()
