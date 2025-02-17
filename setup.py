from setuptools import setup, find_packages

setup(
    name='msna',  
    version='0.1.0',       
    packages=['msna'],  
    install_requires=[  
        'numpy>=1.26,<1.27',
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
