# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='hiive.mdptoolbox',
    version='4.0-b4-dev',
    
    author='Andrew Rollings (originally Steven A. W. Cordwell)',
    author_email='a.rollings@hiive.com',
    url='https://github.com/hiive/hiivemdptoolbox',
    description='Markov Decision Process (MDP) Toolbox',
    long_description='The MDP toolbox provides classes and functions for '
                     'the resolution of discrete-time Markov Decision Processes. The list of '
                     'algorithms that have been implemented includes backwards induction, '
                     'linear programming, policy iteration, q-learning and value iteration '
                     'along with several variations.'
                     ''
                     'Now incorporates visualization code (test)',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3+',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules'],
    platforms=['Any'],
    license='New BSD',

    keywords='mdpviz rl',

    packages=['hiive.mdptoolbox', 'hiive.examples'],

    install_requires=['numpy', 'scipy', 'gym', 'ipython', 'networkx', 'pydot'],
    extras_require={'LP': 'cvxopt'},
    setup_requires=['pytest-runner'],
)
