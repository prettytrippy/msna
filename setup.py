from setuptools import setup, find_packages

setup(
    name='msna',  
    version='0.1.0',       
    packages=['msna'],  
    install_requires=[  
        'numpy>=1.26,<1.27',
        'torch>2.5.0,<=2.5.1',
        'scipy>1.15.0,<=1.15.2',
        'pandas>2.2.0,<=2.2.3',
        'tqdm>4.67.0,<=4.67.1',
        'scikit-learn>1.6.0,<=1.6.1',
        'matplotlib>=3.10.0,<=3.10.0',
    ],
    description='Code for the paper "A Robust Deep Learning Framework for Detecting Bursts in Muscle Sympathetic Nerve Activity"',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown', 
    url='https://github.com/prettytrippy/MSNA', 
    author='Richard Dow',
    author_email='trippdow@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        # 'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6, <3.13'
)
