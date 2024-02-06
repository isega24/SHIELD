Introduction
=================================

SHIELD (Selective Hidden Input Evaluation for Learning Dynamics) is a regularization 
technique that approach aims not only to enhance model interpretability but also to 
directly contribute to overall model quality by enhancing the relationship between 
data and model quality. Specifically, SHIELD adds a regularization term to the
objective function of a model that penalizes the model if it relies too heavily on
a small subset of input features.

Installation
=================================

To use and test SHIELD, we preinstalled the REVEL framework, so you will need to install it first:

.. code-block:: bash

    git clone https://github.com/isega24/ReVel.git
    pip install ./ReVel

Then, you can install SHIELD by running:

.. code-block:: bash

    git clone https://github.com/isega24/SHIELD.git
    pip install ./SHIELD

First example
---------------------------------

After installing SHIELD, you can start using it. For example, on 
`the following jupyter notebook`_ you can see how to use SHIELD to train a model and
evaluate its performance.

.. _the following jupyter notebook: notebooks/SHIELD_example.ipynb