from setuptools import setup, find_packages

package = "unbalanced-metric"
version = "0.1"

setup(
    name=package,
    version=version,
    description="partial distance metric for neural representations",
    packages=find_packages(),
)
