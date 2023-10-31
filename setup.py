from setuptools import setup

setup(
    name='ichthys_harmonics',
    version='0.1',    
    description='A Python package for current predictions from harmonic fits.',
    url='https://github.com/williamedge/ichthys_harmonics',
    author='William Edge',
    author_email='william.edge@uwa.edu.au',
    license='GNU',
    packages=['ichthys_harmonics'],
    install_requires=['numpy',
                      'xarray',
                      'matplotlib',
                      'pandas',
                      'pyTMD'],

    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)
