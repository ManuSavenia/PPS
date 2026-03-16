from setuptools import find_packages, setup

setup(
    name='fingers-quantization-project',
    version='0.2.0',
    description='Fingers dataset feature extraction, MLP training and post-training quantization experiments.',
    author='Manuel',
    packages=find_packages(include=['Fuentes', 'Fuentes.*']),
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'scikit-image',
        'tensorflow',
        'seaborn',
        'ipython',
        'pillow',
        'librosa',
    ],
)