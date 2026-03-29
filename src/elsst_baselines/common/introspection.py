import inspect


def filter_supported_kwargs(callable_obj, candidate_kwargs):
    signature = inspect.signature(callable_obj)
    supported = set(signature.parameters)
    return {key: value for key, value in candidate_kwargs.items() if key in supported}
