import timeit

def time(func, *args):
    a = []
    for e in args:
        a.append(e)
    l = len(a)
    start_time = timeit.default_timer()
    result = func(*args)
    elapsed = timeit.default_timer() - start_time
    return (result, elapsed)


