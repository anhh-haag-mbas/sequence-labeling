import timeit

def time(func, *args):
    start_time = timeit.default_timer()
    result = func(*args)
    elapsed = timeit.default_timer() - start_time
    return (result, elapsed)
