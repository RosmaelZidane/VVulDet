from utils import preprocessdata as prep

"""Run the prepare file"""

def prepared():
    prep.CVEFixes()
    prep.get_dep_add_lines_CVEFixes()
    return 

if __name__ == "__main__":
    prepared()  