def lazy_module(module_map, package):

    def __getattr__(name: str):

        if name not in module_map:
            raise AttributeError(name)

        import importlib
        module = importlib.import_module(
            module_map[name],
            package=package,
        )

        obj = getattr(module, name)
        globals()[name] = obj
        return obj

    return __getattr__
