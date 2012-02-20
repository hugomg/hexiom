Using an industrial-strength SAT solver to solve the Hexiom puzzle
==================================================================

Recently there was some talk on Reddit
([here](http://www.reddit.com/r/programming/comments/p54v2/solving_hexiom_perhaps_you_can_help/) and
[here](http://www.reddit.com/r/programming/comments/pjx99/solving_hexiom_using_constraints/)) about using custom algorithms to play
[Hexiom](http://www.kongregate.com/games/Moonkey/hexiom), a cute little Flash puzzle game from Kongregate.

The original submitter has published some solutions using traditional backtracking and exaustive search on [his repository](https://github.com/slowfrog/hexiom), but I wandered
if I could do better then him by leveraging a Boolean Satisfability solver. In particular, one of the problem sets (level 38) would take hours to complete and I wanted to see if I could do better then that.

Why bother with a SAT solver?
-----------------------------

Why should I complicate things with a SAT solver instead of tinkering with an existing backtracking solution? My motivations are basically that

* Modern SAT solvers have very optimized and smart brute forcing engines. They can do many tricks that most people do not know about or do not go through the trouble of implementing most of the time:
   * Non chronological backtracking (can go back multiple levels at once)
   * Efficient data structures / watched literals (backtracking is O(1))
   * Clause learning (avoids searching the same space multiple times)
   * Random and aggressive restarts (avoids staying too long in a dead end)
   * ...and much more!

* SAT solver problem input is declarative so it is easy to add new rules and solving strategies without needing to rewrite the tricky backtracking innards.
   * This will be particularly important when we get to the symmetry-breaking optimizations.

Or in other words, SAT solvers are very fast and let me easily do things I would usually not try to do using a more traditional approach.

My final results
----------------

In terms of speed, I **managed to solve the case that used to take hours in took hours in just over one minute**, while also still taking just a couple of seconds for the other easy problems.

In terms of programming and algorithms, the good part is precisely that **I didn't have to do anything very special**.
Besides encoding the problem using SAT, **the underlying algorithm is still exaustive backtracking, without any Hexiom-specific heuristics added<sup>1</sup>.**

<sup>1</sup> (Well, one might try to count the symmetry-breaking as Hexiom-specific but the overall techinique is still pretty general...)

-----------------------------

How this all works
==================

The puzzle
----------

A Hexiom puzzle instance consists of an hexagonal playing field and a set of numbered, hexagonal, tiles. The objective consists in placing the tiles in slots on the board in such a way that the number on each tile corresponds to the amount of neighboring tiles it has.

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

The Boolean Satisfiability Problem
----------------------------------

SAT solvers, as the name indicates, solve the [boolean satisfiability](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem) problem. This problem consists of determining, given a set of boolean variables and a set of propositional predicates over these variables, whether there is a true-false assignment of the variables that satisfies all the predicates.

For example, given variables `x`, `y`, and predicates

    1) (NOT x) OR y
    2) y OR z
    3) (NOT y)  OR (NOT z) 
    4) x

We can produce the assignment {X=1, Y=1, Z=0} that satisfies all 4 clauses.

However if where to add the 5-th clause

    5) (NOT x) OR z
    
then there would be no solutions.

Modeling Hexiom using as a SAT instance.
----------------------------------------

I used the following variable encodings to model the problem domain:

* O<sub>m</sub> := The m-th slot of the board has a tile on it.
* P<sub>m,n</sub> := There is a n-valued tile on the m-th slot on the board
* N<sub>m,k</sub> := The m-th slot has k tiles neighboring it.
* T<sub>n,k</sub> := There are k n-valued tiles placed on the board.
* ... and other helper variables for the cardinality constraints, etc.

From this on its a matter of writing the predicates:

* General predicates to define the variables as we intend them to be. They are mostly shared by all Hexiom instances of the same hegagon side length:
   * A cardinality constraint to say that a board slot is either empty or has a single slot on it
   * Cardinality constraints to define N<sub>m,k</sub>
   * Cardinality constraints to define T<sub>n,k</sub>

* Level-dependant predicates, to describe the specific Hexiom level:
   * Set T<sub>n,k</sub> according to the actual number ot tiles available.
   * Set P<sub>m,n</sub> and O<sub>m</sub> for slots that come with preset values that cannot be changed.

The only hard bit up to here is the cardinality constraints. For the small case (the rule for only a tile per slot) I " brute-forced" (O(n^2)) it and made a rule for each pair of variables saying at least one of them must be false.

For the other cardinality constraints, I used an unary encoding, with helper variables such as `Nps(m, k, i) := There are at least k occupied tiles among the first i neighbors of tile m`. This gives a compact encoding, unlike the naïve version that lists all exponentialy-many possibilities.

First results
-------------

With the initial description of the problem I already was already able to achieve results similar to those from the manually written backtracking search: all levels could be solved in under a couple of seconds, except for level 38, which took around half an hour to solve.

Breaking symmetries
-------------------

The Hexiom instance that took the longest to solve was highly symmetrical so I suspected that the solver (and the backtracking-based approach) were wasting many cycles trying the same things multiple times (but in mirrored ways). I added *symmetry-breaking predicates* to rule out equivalent solutions from the search space.

Hexagonal symmetries can be boiled down to the following 12 rotations and reflections (the 12 comes from 6 possible rotations times 2 for either doing a reflection or not):


    Rotations

     1 2    6 1    5 6
    6 0 3  5 0 2  4 0 1  ... and 3 more ...
     5 4    4 3    4 2  

    Reflections

     1 6    6 5    5 4
    2 0 5  1 0 4  6 0 3  ... and 3 more ...
     3 4    2 3    1 2
 
The trick behind writing a symmetry-breaking predicate is that if we arrange the variables corresponding to a solution in one of the permutations

    VX = O(1), P(1, 1..6), O(2), P(2, 1..6), ..., O(6), P(6, 1..6)

And the corresponding variables after a symmetric transformation (say a clockwise rotation of 1/6th of a turn)

    VY = O(2), P(2, 1..6), O(3), P(3, 1..6), ..., O(1), P(1, 1..6)

It is clear that given a satisfying assignment in VX we can find a symmetric assignment via VY. By imposing an arbitrary total order on these assignments we can force it so that only one of them is allowed (saving the SAT solver from exploring its reflections multiple times). The standard way to do this is to think of VX and VY as bit vectors representing integers and then write predicates that state the equivalent of `VX <= VY`.

----------------------

How do I run this thing then?
=============================

    python hexiom_solve.py NN
    
Where NN is a number from 0 to 40 standing for the number of the level you want to solve. It will use an input file from the levels folder I copied from Slowfrog's project.

Where do I get a SAT solver?
=============================

I am including copies of some SAT solver executables in this repo but I am not sure they will work on other computers and platforms.

In any case, the hexiom solver was designed to be able to handle any SAT solver that uses the relatively standard DIMACS input and output formats.  You can find a good selection of state of the art solvers in the websites of the annual [SAT Competition](http://satcompetition.org/).

[Page of the SAT competition with links to the solver websites](http://www.cril.univ-artois.fr/SAT11/)

Here are some of the best preforming SAT solvers from last year if you want to check them out:

* [Glucose](http://www.lri.fr/~simon/?page=glucose) - [Source link](http://www.lri.fr/~simon/downloads/glucose-2-compet.tgz)
* [Cryptominisat](http://www.msoos.org/cryptominisat2/) - [Source Link](https://gforge.inria.fr/frs/download.php/30138/cryptominisat-2.9.2.tar.gz)
* [Lingeling](http://fmv.jku.at/lingeling/) - [Source Link](http://fmv.jku.at/lingeling/lingeling-587f-4882048-110513.tar.gz)

As far as compiling goes, all the solvers I linked to are written in C or C++ and all of them come with an easy to use makefile or build script.
