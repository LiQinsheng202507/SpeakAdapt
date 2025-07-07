from setuptools import setup, find_packages

setup(
    name='cast_hcas_speech',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='CAST + HCAS: Context-Aware Deep Learning Framework for Robust Speech Recognition',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourname/CAST-HCAS-SpeechRecognition',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.0.0',
        'torchaudio>=2.0.0',
        'librosa>=0.10.0',
        'soundfile>=0.12.1',
        'numpy>=1.23.0',
        'pandas>=1.5.0',
        'matplotlib>=3.7.1',
        'seaborn>=0.12.2',
        'tqdm>=4.66.0',
        'scikit-learn>=1.3.0',
        'jiwer>=3.0.1',
        'sacrebleu>=2.3.1',
        'transformers>=4.38.0',
        'sentencepiece>=0.1.99',
        'PyYAML>=6.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'cast-hcas-train=main:main'
        ]
    },
    include_package_data=True,
)
