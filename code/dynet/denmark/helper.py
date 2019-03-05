import timeit

def time(func, *args):
    a = []
    for e in args:
        a.append(e)
    l = len(a)
    start_time = timeit.default_timer()
    result = None
    if l == 0:
        result = func()
    elif l == 1:
        result = func(a[0])
    elif l == 2:
        result = func(a[0], a[1])
    elif l == 3:
        result = func(a[0], a[1], a[2])
    elif l ==4:
        result = func(a[0], a[1], a[2], a[3])
    elapsed = timeit.default_timer() - start_time
    return (result, elapsed)


