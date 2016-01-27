.. dsge documentation master file, created by
   sphinx-quickstart on Mon Jan 25 12:23:32 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. include:: ../../README

Contents:

========
Overview
========
Here we give an overview of the two major objects that the package
provides.  The first is :any:`DSGE.DSGE`, a container for the symbolic
representation of the DSGE model.  :any:`DSGE.DSGE` inherits from a
dictionary, but the best (only) way to create a DSGE object is through
the static method :any:`DSGE.DSGE.read`

Reading a Model File
~~~~~~~~~~~~~~~~~~~~
Models are stored in a human-readable markup language call
`YAML<http://yaml.org>` (YAML).  YAML stores information in plain
text, structured using blocks (spacing), dashes (entries within a
block), and colons (for key-value mappings.)

The structure of the minimally acceptable file looks like:

- **declarations** :

  - **name** : a string with the name of the model
  - **variables** : a list of the model variables
  - **shocks** : a list of the exogenous shocks in the model
  - **parameters** : a list of the model parameters

Consider the simple model which is defined by the single equation,

.. math::

   y_t &=& \beta E_t[y_{t+1}] + x_t \\
   x_t &=& \rho x_{t-1} + \epsilon_t, \quad \epsilon\sim N(0,\sigma^2)


We fill out the declarations section in this in the YAML file as
follows.

.. literalinclude:: ../../dsge/examples/simple-model/simple_model.yaml
   :lines: 1-5
   :language: yaml

In the next section of the YAML file we define introduce the **equations**
block in which we define the model as a list of equations.  For time t
variable $x$ use :code:`x` for the current variable, :code:`x(-n)` for
the value at time $t-n$ and :code:`x(+n)` for the expectation of the
time $t+n$ value taken at time $t$.

.. literalinclude:: ../../dsge/examples/simple-model/simple_model.yaml
   :lines: 6-8
   :language: yaml

The final required block is the **calibration** block, which defines
parameter values for the parameters and covariance matrix.

.. literalinclude:: ../../dsge/examples/simple-model/simple_model.yaml
   :lines: 9-15
   :language: yaml

The initialize a DSGE model object, use the static method :any:`read`,
an example of which is given below.

.. ipython::

   In [1]: import dsge

   In [2]: my_model = dsge.DSGE.DSGE.read('../dsge/examples/simple-model/simple_model.yaml')

   In [3]: print my_model.equations

   In [4]: print my_model.variables

   In [5]: print my_model.shocks

   In [6]: print my_model.parameters

Richer Model Declarations
~~~~~~~~~~~~~~~~~~~~~~~~~
Given the emphasis on estimating models, one can add options to the
YAML to define an estimatable model.  The first thing is to define the
**observables** in the model declaration section.

.. literalinclude:: ../../dsge/examples/simple-model/simple_model_est.yaml
   :lines: 1-6
   :emphasize-lines: 6
   :language: yaml

Next we need to modify the **equations** section to include both the
model equations and a dictionary mapping the observables into the
model objects.

.. literalinclude:: ../../dsge/examples/simple-model/simple_model_est.yaml
   :lines: 7-11
   :emphasize-lines: 2,4-5
   :language: yaml

Finally, we include an addition block in our YAML file for the
**estimation** of the model.  The **data** subblock points to a text
file which contains the data for the estimation.  It should just be
plain text, no headers or index columns.  The **prior** subblock
contains a dictionary defining the prior distribution for each
parameter.  The prior is defined as a list with three elements: the
name of the prior distribution, two hyperparameters defining the moments
of the distribution.  Valid prior distributions are
:code:`beta`, :code:`uniform`, :code:`gamma`, :code:`normal`, and
:code:`inv_gamma`.  The two hyperparameters for the :code:`beta`,
:code:`gamma`, :code:`normal` define the mean and standard
deviation, repsectively.  For the :code:`uniform` distribution, these
two hyperparameters are the lower and upper bound, while for the
:code:`inv_gamma` distribution they define the s^2 and nu, in
Zellner's (1971) notation.

.. literalinclude:: ../../dsge/examples/simple-model/simple_model_est.yaml
   :lines: 19-
   :language: yaml

Compiling Models
----------------

To turn the symbolic representation of the model into a
:any:`LinearDSGEModel`, we use the method :any:`compile_model`.

.. ipython::

   In [1]: my_model = dsge.DSGE.DSGE.read('../dsge/examples/simple-model/simple_model_est.yaml')

   In [2]: compiled_model = my_model.compile_model()

The compiled model is the Linear State Space representation of the
DSGE model, with an additional method :any:`solve_LRE`, which solves
the linear rational expectations system using GENSYS [see Sims
(2002)].

.. ipython::

   In [1]: p0 = my_model.p0()

   In [2]: print p0

   In [3]: TT, RR, RC = compiled_model.solve_LRE(p0)

   In [4]: print TT

   In [5]: print RR


Since the :any:`LinearDSGEModel` inherits from a
:any:`StateSpaceModel`, we can also get the state space
representation, compute impulse responses, simulate from the model,
evaluate the likelihood functon and more.

.. ipython::

   In [1]: TT,RR,QQ,DD,ZZ,HH = compiled_model.system_matrices(p0)

   In [2]: irfs = compiled_model.impulse_response(p0, h=10)

   In [3]: print irfs['e']

   In [4]: y = compiled_model.simulate(p0)

   In [5]: print compiled_model.log_lik(p0,y=y)

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
