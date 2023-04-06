MAX_NUM_LOOPS = 100


class InfiniteLoopError(Exception):
    """
    An error raised when a loop runs for too many iterations.
    """
    pass


class InvalidAnnotationError(Exception):
    """
    An error raised when an annotation is invalid.
    """
    pass


class InvalidSample(Exception):
    """
    An error raised when a sample is invalid.
    """
    pass
