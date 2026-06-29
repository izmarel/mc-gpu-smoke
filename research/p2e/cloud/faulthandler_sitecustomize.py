import faulthandler, signal, sys
faulthandler.enable()
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
except Exception as e:
    sys.stderr.write("[sitecustomize] register failed: %r\n" % e)
try:
    import faiss; faiss.omp_set_num_threads(1)
except Exception as e:
    sys.stderr.write("[sitecustomize] faiss thread set failed: %r\n" % e)
sys.stderr.write("[sitecustomize] faulthandler(SIGUSR1)+faiss1thread armed\n"); sys.stderr.flush()
