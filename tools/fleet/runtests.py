import importlib.util,sys,traceback,inspect
sys.path.insert(0,".")
fails=0;n=0
for path in sys.argv[1:]:
    spec=importlib.util.spec_from_file_location("t",path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for name,f in inspect.getmembers(m,inspect.isfunction):
        if name.startswith("test_") and f.__module__=="t" and not inspect.signature(f).parameters:
            n+=1
            try: f()
            except Exception: fails+=1; print("FAIL",path,name); traceback.print_exc(limit=3)
print(f"{n-fails}/{n} passed")
