# A Simple Optimization Program for CIE6036 Project #
This is a repository for the solution to our project of the course CIE6036.
## Package Description ##
- The `/data` directory is for the storage of our data in the form of .csv file
- The `/result` directory stores the output of the optimization program
- The `method_cplex.py` use **cplex** to solve a linear optimization problem
- The result of `method_cplex.py` is saved in `/result/cplex_optimization_solution.csv`
## Program Setup ##
- Create a virtual python environment using conda: `conda env create -f requirement.yaml`
- If you are using an IDE, set the project interpreter as `opt`, which is the environment you just created
- If you are just using the console, you can just type `conda activate opt`
- Run an executable .py file to get the result