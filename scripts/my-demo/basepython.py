def example_decorator(func):
    def wrapper(*args, **kwargs):
        print("Arguments passed to the function:", args)
        print("Keyword arguments passed to the function:", kwargs)
        return func(*args, **kwargs)
    return wrapper

@example_decorator
def my_function(a, b, c, **kwargs):
    print(a, b, c)
    print("Additional keyword arguments:", kwargs)

my_function(1, 2, 3, d=4, e=5)