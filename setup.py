from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_requirements():
    path = ROOT / 'requirements' / 'runtime.txt'
    return [
        line.strip()
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.startswith('#')
    ]


version_scope = {}
exec((ROOT / 'openpsg' / 'version.py').read_text(encoding='utf-8'),
     version_scope)

setup(
    name='srp-mtmnet',
    version=version_scope['__version__'],
    description='Multimodal PSGFormer for joint segmentation and relation prediction',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=('configs', 'tools')),
    install_requires=read_requirements(),
    python_requires='>=3.8,<3.9',
    license='Apache-2.0',
    zip_safe=False,
)
