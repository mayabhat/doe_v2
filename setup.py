# Copyright 2022 Maya Bhat
# (see accompanying license files for details).
from setuptools import setup

setup(name='doe',
      version='0.0.1',
      description='doe, model, and analysis',
      url='',
      maintainer='Maya Bhat',
      maintainer_email='mayabhat@andrew.cmu.edu',
      license='GPL',
      platforms=['linux'],
      packages=['doe'],
      setup_requires=[],
      data_files=[],
      install_requires=['autograd', 'pyDOE2', 'sklearn', 'matplotlib', 'numpy', 'pycse', 'pandas', 'xlrd', 'openpyxl'],
      long_description='''A module to perform Design of Experiments design, model building, and statistical analysis
      ''')

# (shell-command "python setup.py register") to setup user
# to push to pypi - (shell-command "python setup.py sdist upload")


# Set TWINE_USERNAME and TWINE_PASSWORD in .bashrc
# python setup.py sdist bdist_wheel
# twine upload dist/*
