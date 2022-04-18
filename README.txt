Caleb Goertel - Assignment 4

To run this code, it must first be checked out via git or downloaded from:

https://github.com/cgu11/gtmlassignment4

where it will be publicly available until the grade is returned. 


Files forest.py, frozen_lake.py, models.py are scripts housing 
helper functions and do not need to be run directly, they will be imported by the notebooks.

The hiive-mdptoolbox repository must be downloaded directly from git instead of pip, since I needed a version unavailable through pip.

To get it, run 

git clone https://github.com/hiive/hiivemdptoolbox.git

to check out the repository, then when in the same directory as the hiivemdptoolbox folder it created, run

pip install hiivemdptoolbox/

it's important to have the trailing slash or else it will install from PYPI instead of the local folder.


From there, with requisite packages installed (matplotlib, numpy, pandas, jupyter notebooks, gym), opening
the python notebook and running it straight through will reproduce the models, outputs and charts discussed in the report.
