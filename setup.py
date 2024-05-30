from setuptools import find_packages,setup
from typing import List



def get_requirements(file_path:str)->list[str]:
    '''
    This function will return the list of required libraires

    '''
    requirements = []

    HYPEN_E_DOT = '-e .'

    with open(file_path) as file:
        requirements = file.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements



setup(name = 'Potability Prediction',
      version = '0.0.1',
      author = 'Pranav',
      author_email='pranavp1712@gmail.com',
      packages= find_packages(),


      install_requires = get_requirements('requirements.txt'))
