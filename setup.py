from setuptools import setup, find_packages

setup(
    name="msna",
    version="0.1.0",
    packages=find_packages(where="."),
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "tqdm",
    ],
    description='Code for the paper "A Robust Deep Learning Framework for Detecting Bursts in Muscle Sympathetic Nerve Activity"',
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/prettytrippy/msna',
    author='Tripp Dow',
    author_email='trippdow@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6, <3.13',
)
