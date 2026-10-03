"""PyInstaller entry point for NiveshRL.exe (keeps the package importable as ``niveshrl``)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()          # the pipeline worker process re-enters here
    from niveshrl.desktop.app import main
    sys.exit(main())
