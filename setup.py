import setuptools


## ROS2 simlink didn't like this.
# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="kincalib",
    version="0.0.0",
    author="Juan Antonio Barragan",
    author_email="jbarrag3@jh.edu",
    description="Da vinci kinematic calibration project",
    long_description="",#long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=["rich", "click"],
    include_package_data=True,
    python_requires=">=3.7",
)
