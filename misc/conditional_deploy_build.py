import sys, os, subprocess

if os.getenv("APPVEYOR_REPO_TAG") == "true":
    proc = subprocess.Popen(['python','-m', 'twine', 'upload','--skip-existing', 'wheelhouse/*'])
    proc.communicate() # wait for it to terminate

    # forward the return code
    code = proc.returncode
    sys.exit(code)

else:
    print("not deploying, no appveyor repo tag present")
