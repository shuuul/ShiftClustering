import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup, Extension, find_packages

ext_modules = [
    Extension("shiftclustering._meanshiftpp",
              sources=["shiftclustering/_meanshiftpp.pyx"],
              language="c++",
              include_dirs=[numpy.get_include(), 'lib']),
    Extension("shiftclustering._gridshift",
              sources=["shiftclustering/_gridshift.pyx"],
              language="c++",
              include_dirs=[numpy.get_include(), 'lib'])
]

setup(
    name='ShiftClustering',
    version='0.1.2',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules),
    author='Shu Li',
    author_email='zetarylee@gmail.com',
    description='A package for fast mean shift clustering algorithms',
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    packages=find_packages(),
)
