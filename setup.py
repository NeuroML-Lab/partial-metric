from setuptools import setup, find_packages

package = "unbalanced-metric"
version = "0.1"

setup(
    name=package,
    version=version,
    description="unbalanced distance metric for neural representations",
    packages=find_packages(),
    #package_dir={'': 'src'},
    #packages=find_packages(where="src"),
)
