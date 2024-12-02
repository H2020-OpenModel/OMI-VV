#!/usr/bin/env python
import aiida
from aiida import orm, engine
from execflow.workchains.declarative_chain import DeclarativeChain
import sys
import os
from pathlib import Path
from vv.vv import run_vv  # Import your VV script

# AiiDA setup
aiida.load_profile()

if __name__ == "__main__":
    workflow = sys.argv[1]

    # Step 1: Run AiiDA calculation
    all_inputs = {"workchain_specification": orm.SinglefileData(os.path.abspath(workflow))}
    engine.run(DeclarativeChain, **all_inputs)

    # Step 2: Run VV step
    run_vv(workflow)  # Execute VV using the same YAML file

