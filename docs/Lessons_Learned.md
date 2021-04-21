# New experiences and lessons learned

Setting up Docker
Using Pip-compile
Using Pickle to save models
Chance to explore how others would use this code, and helped me learn how to do it right the first time.

## Process improvements

Need to have a uniform dev project workflow
Use ONNX (I wanted to use it and then got weird errors that told me I needed to learn a lot more) 
Data Version Control
Testing framework

## Useful Commands

To deal with finding your own packages in a notebook:

~~~python
import sys
sys.path.append('..')
~~~

To install your own package
pip install -e .

python3.8 -m piptools compile --allow-unsafe
