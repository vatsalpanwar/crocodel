from setuptools import setup

setup(
    name='crocodel',
    version='0.0.1',    
    description='A Python package for high-resolution cross-correlation spectroscopy retrievals of exoplanet atmospheres.',
    url='https://github.com/vatsalpanwar/crocodel',
    author='Vatsal Panwar',
    author_email='panvatsal@gmail.com',
    license='MIT License',
    packages=['crocodel'],
    install_requires=['numpy', 'astropy', 'scipy', 
                      'matplotlib', 'tqdm', 'yaml',                    
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
)