from setuptools import setup, find_packages

# Use requirements text to manage the dependencies.
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Knowledgeable-Stories',
    version='0.0.1',
    packages=find_packages(),
    install_requires=requirements,
    url='',
    license='',
    author='David Wilmot',
    author_email='david.wilmot@ed.ac.uk',
    description='2nd iteration of the story models to incorporate multitask learning and richer semantic knowledgebase, event and character based understanding.',
)
