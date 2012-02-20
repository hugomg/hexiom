Solving the Hexiom Puzzle with industrial-strength SAT Solvers
==============================================================

Hexiom is one of those little Flash puzzle games. (You can play it [here](Flash http://www.kongregate.com/games/Moonkey/hexiom)).

There has recently been some talk on Reddit ([here](http://www.reddit.com/r/programming/comments/p54v2/solving_hexiom_perhaps_you_can_help/) and [here](http://www.reddit.com/r/programming/comments/pjx99/solving_hexiom_using_constraints/)) on using a computer algorithm to solve the game (see slowfrog's [github repo](https://github.com/slowfrog/hexiom)).

Slowfrog used traditional backtracking and brute force tachniques to solve the puzzle and got good results, except on the level 38, where the solver supposedly only found a solution after 2 hours of work (more on this bit latter...).

So why SAT solvers?
-------------------

Backtracking and exaustive search are simple algorithms that anyone can pull off without too much trouble. Why would I want to complicate things with a SAT solver?

* Modern SAT solvers have very optimized and smart brute forcing engines. The tricks they use are much more advanced then regular backtracking
   * Non chronological backtracking (can go back multiple levels at once)
   * Efficient data structures (backtracking is O(1))
   * Clause learning (avoids searching the same space multiple times)
   * Random and aggressive restarts (avoids staying too long in a dead end)

* SAT solvers let me describe the problem declaratively (kind of like Prolog, but super efficient instead).

* It is also possible to add optimizations via extra predicates without having to rewrite or overhaul the complicated inner backtracking algorithm. (This was very important for the important symmetry-breaking optimizations)

My final results
----------------

I managed to also solve most of the levels in just a couple of seconds but I **also managed to solve the case that took hours in just over one minute!**. The best part is that all of this was still just doing a smart brute force search - I did not have to use or invent any Hexiom-specific algorithms or techniques.*

*(Well, I had to think a bit for the symmetry part latter on but even this is kind of a standard technique...)

-----------------------------

The longer explanation
=========

The game
--------

A Hexiom puzzle instance consists of an hexagonal playing field and a set of numbered (hexagonal) tiles. The objective consists in placing the tiles in the board in such a way that the number on each tile corresponds to the amount of other neighboring tiles it has.

For example, level 8 presents the following initial placement of tiles (the dots represent empty board slots):

      4 1 4
     1 . . 1
    4 . 6 . 4
     1 . . 1
      4 1 4
 
And has the following solution

      1 . 1
     . 4 4 .
    1 4 6 4 1
     . 4 4 .
      1 . 1

Note how each **1** is surrounded by one other number, how each **4** is surrounded by four other numbers and how the **6** has a full set of 6 neighboring tiles around it.

Modeling Hexiom using as boolean satisfiability problem.
-----------

SAT solvers receive inputs encoded as a [boolean satisfiability](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem) problem that consists of a set of boolean variables (that can be true or false) and a list of propositional formulas<sup>1</sup> (predicates) that these variables should obey. The SAT solver will then search for an assignment of variables that satisfies all the given restrictions.

I used the following variable encodings to model the problem domain:

* O<sub>m</sub> := The m-th slot of the board has a tile on it.
* P<sub>m,n</sub> := There is a n-valued tile on the m-th slot on the board
* N<sub>m,k</sub> := The m-th slot has k tiles neighboring it.
* T<sub>n,k</sub> := There are k n-valued tiles placed on the board.
* ... and other helper variables for the cardinality constraints, etc.

From this on its a matter of writing predicates:

* General predicates (that give the variables the meaning we intend them to):
   * A cardinality constraint to say that a board slot is either empty or has a single slot on it
   * Cardinality constraints to define N<sub>m,k</sub>
   * Cardinality constraints to define T<sub>n,k</sub>

* Level-dependant predicates (that describe the input):
   * Set T<sub>n,k</sub> according to the actual arrangement of tiles.
   * Set P<sub>m,n</sub> or O<sub>m</sub> for slots that come with preset values that cannot be changed.

The only hard bit up to here is the cardinality constraints. For the small case (only a tile per slot) I just" brute-forced" (O(n^2)) it and made a rule for each pair of variables saying at least one of them must be false.

For the other cardinality constraints, I used a unary encoding, with helper variables such as `Nps(m, k, i) := There are at least k occupied tiles among the first i neighbors of tile m`. This gives a quadratic encoding of the sum instead of doing the naive approach of listing all the possibilities.

First results
-------------

With the initial description of the problem I already was already able to achieve results similar to those from the manually written backtracking search: all levels could be solved in under a couple of seconds, except for level 38, that took some 27 minutes to finish.

Breaking symmetry
-----------------

The problem that took the longest to solve was highly symmetrical so I suspected that the solver (and the backtracking-based approach) were wasting many cycles trying the same thing multiple times (but in mirrored ways). I added *symmetry-breaking predicates* to rule out equivalent solutions from the search space.

Hexagonal symmetries can be boiled down to the following 12 rotations and reflections (the 12 comes from 6 possible rotations times 2 for either doing a reflection or not):


    Rotations

     1 2    6 1    5 6
    6 0 3  5 0 2  4 0 1  ... and 3 more ...
     5 4    4 3    4 2  

    Reflections

     1 6    6 5    5 4
    2 0 5  1 0 4  6 0 3  ... and 3 more ...
     3 4    2 3    1 2
 
The trick behind writing a symmetry breaking predicate is that if we arrange the variables corresponding to a solution in one of the permutations

    VX = O(1), P(1, 1..6), O(2), P(2, 1..6), ..., O(6), P(6, 1..6)

And the corresponding variables after a symmetric transformation (say a clockwise rotation of 1/6th of a turn)

    VY = O(2), P(2, 1..6), O(3), P(3, 1..6), ..., O(1), P(1, 1..6)

It is clear that given a satisfying assignment in VX we can find a symmetric assignment via VY. By imposing an arbitrary total order on these assignments we can force it so that only one of them is allowed (saving the SAT solver from exploring its reflections multiple times). The standard way to do this is to think of VX and VY as bit vectors representing integers and then write predicates that state the equivalent of `VX <= VY`.

-------

1 - Actually, most SAT solvers demand that the input predicates be in clausal normal form. That is, each predicate is the OR of a list of literals and each literal is either a variable or its negation. Having the input in CNF form is not a big restriction (there are straightforward ways to efficiently convert things into it) but allows for very efficient algorithms and data structures to be used under the hood.

----------------------

How do I run this thing then?
=============================

    python hexiom_solver NN
    
Where NN is a number from 0 to 40 standing for the number of the level you want to solve. It will use an input file from the levels folder I copied from Slowfrog's project.

Where do I get a SAT solver?
=============================

A good place to get hot SAT solvers is the website of the annual [SAT Competition](http://satcompetition.org/).

[Page of the SAT competition with links to the solver websites](http://www.cril.univ-artois.fr/SAT11/)

Some of the best preforming SAT solvers from last year:

* [Glucose](http://www.lri.fr/~simon/?page=glucose) - [Source link](http://www.lri.fr/~simon/downloads/glucose-2-compet.tgz)
* [Cryptominisat](http://www.msoos.org/cryptominisat2/) - [Source Link](https://gforge.inria.fr/frs/download.php/30138/cryptominisat-2.9.2.tar.gz)
* [Lingeling](http://fmv.jku.at/lingeling/) - [Source Link](http://fmv.jku.at/lingeling/lingeling-587f-4882048-110513.tar.gz)

Most of them use similar, standardized, input and output formats and it is easy to switch them around (see the `hexiom_config.py` file). 
As far as compiling goes, the solvers I linked to are all written in C or C++ and come with easy to use makefiles or build scripts. (I didn't try to package a working executable here already because I often fail terribly at making them be cross platform...)
