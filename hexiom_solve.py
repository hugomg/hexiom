#! /usr/bin/python
# - * - coding:utf-8 - * -

############################################
# CONFIGURATION
############################################

from hexiom_config import *

############################################

####
# Helper functions
###

def sign(n):
    if n < 0:
        return -1
    elif n == 0: 
        return 0
    else: #n > 0
        return +1

def enumerate1(xs):
    ''' (x1,x2...) -> ((1,x1), (2,x2),...) '''
    for (i,x) in enumerate(xs):
        yield (i+1, x)

def concat(xss):
    ''' [[a]] -> [a] ''' 
    return [x for xs in xss for x in xs]

def permutations(xs):
    def go(i):
        if i >= len(xs):
            yield []
        else:
            for x in xs[i]:
                for p in go(i+1):
                    yield [x] + p
    return go(0)

def assoc_list(xs):
    return list(enumerate(xs))

def assoc_get(x, xs):
    for (k,v) in xs:
        if k == x:
            return v
    return None

##
# Building rules
#

def size_of_dims(dims):
    s = 1
    for (minv, maxv) in dims:
        if minv <= maxv:
            s *= (maxv - minv + 1)
        else:
            s *= 0
    return s

def dim_multipliers(dims):
    ''' mult[i] = size_of(dims[i+1:])'''
    
    multipliers = [None] * len(dims)
    multipliers[-1] = 1
            
    for i in range(len(dims)-2, -1, -1):
        (minv, maxv) = dims[i+1]
        multipliers[i] = \
            multipliers[i+1] * \
            (maxv - minv + 1)

    return multipliers

class InvalidVariable(Exception):
    pass


class RuleSet(object):
    
    def __init__(self):
        self.vsets = []
        self.vsets_by_name = {}
        self.next_free_variable = 1

    class Variable(object):
        def __init__(self, vset, vs):
            self.vset = vset
            self.vs = vs

        def __str__(self):
            return self.vset.name + '(' + ' '.join(map(str,self.vs)) + ')'

        def __eq__(self, b):
            return (self.vset is b.vset) and (self.vs == b.vs)

        def offsetOf(self):
            return self.toInt() - self.vset.first_variable

        def toInt(self):
            return self.vset.indexOfVariable(self)

    class VariableSet(object):
        def __init__(self, ruleset, name, dims):
            self.name = name

            self.ndims = len(dims)
            self.dims = dims
            
            self.size = size_of_dims(dims)

            self.multipliers = dim_multipliers(dims)

            self.ruleset = ruleset
            ruleset.vsets.append(self)

            if ruleset.vsets_by_name.get(name):
                raise Exception('repeated name %s'%name)
            ruleset.vsets_by_name[name] = self
            
            self.first_variable = self.ruleset.next_free_variable
            self.ruleset.next_free_variable += self.size

        def __call__(self, *vs):
            if self.ndims != len(vs):
                raise InvalidVariable(
                    'Expected %d dimensions, got %d'%(
                        self.ndims, len(vs))
                )

            for (val, (min_val, max_val)) in zip(vs, self.dims):
                if not (min_val <= val <= max_val):
                    raise InvalidVariable(
                        'Variable out of bounds:', self.name, vs, self.dims
                    )
                    
            return Lit(+1, RuleSet.Variable(self, vs) )
        
        def contains(self, n):
            return 0 <= (n-self.first_variable) < self.size

        def indexOfVariable(self, var):
            if var.vset is not self:
                raise Exception('Converting variable at the wrong place')

            return self.first_variable + sum([
                (val - minv)*mult
                for (val, (minv, maxv), mult) in
                zip(var.vs, self.dims, self.multipliers)
            ])
            
        def valuesOfIndex(self, n):
            offset = n - self.first_variable

            vs = []
            for mult in self.multipliers:
                vs.append(offset//mult)
                offset = offset%mult

            return tuple(vs)
   
    def VarSet(self, name, dims):
        return RuleSet.VariableSet(self, name, dims)

    def print_cnf_file(self, formulation, fil):
        ruleset = formulation.ruleset
        clauses = formulation.clauses

        with open(NAMED_CNF_INPUT_FILE, 'w') as dbg:
        
            print >> fil, 'p cnf', ruleset.next_free_variable -1 , len(clauses)
            for (i,clause) in enumerate(clauses):
                for lit in clause:
                    print >> dbg, lit,
                print >> dbg
                    
                intClause = [lit.sign*lit.var.toInt() for lit in clause]
                for lit in intClause:
                    print >> fil, lit,
                print >> fil, '0'

            nvars = self.next_free_variable - 1

    def get_varset_by_name(self, name):
        return self.vsets_by_name.get(name)

    def get_varset_by_variable(self, n):
        beg = 0
        end = len(self.vsets)
        while(beg < end):
            m = (beg + end) // 2
            vset = self.vsets[m]
            if n < vset.first_variable:
                end = m
            elif n >= vset.first_variable + vset.size:
                beg = m+1
            else:
                return vset
        return None

    def cnfVarToLit(self, lit_n):
        sgn = (1 if lit_n >= 0 else -1)
        n   = abs(lit_n)
        varset = self.get_varset_by_variable(n)
        return Lit(
            sgn,
            RuleSet.Variable(varset, varset.valuesOfIndex(n))
        )
#####
# Logical connectives
####

class Logic(object):
    def __pos__(self):
        return self

class Lit(Logic):
    def __init__(self, sign, var):
       self.sign = sign
       self.var = var
       
    def __str__(self):
        sgn = ('+' if self.sign > 0 else '-')
        return sgn + str(self.var)
 
    def __neg__(self):
        return Lit(-self.sign, self.var)

    def to_cnf(self):
        return [[self]]

class And(Logic):
    def __init__(self, cs):
        self.cs = cs

    def __neg__(self):
        return Or([-c for c in self.cs])

    def to_cnf(self):
        return concat(c.to_cnf() for c in self.cs)

class Or(Logic):
    def __init__(self, cs):
        self.cs = cs

    def __neg__(self):
        return And([-c for c in self.cs])

    def to_cnf(self):
        return map(concat, permutations( [c.to_cnf() for c in self.cs] ) )

def Implies(a,b):
    return Or([-a, +b])

def Equivalent(a,b):
    return And([
        Implies(a,b),
        Implies(b,a)
    ])

def BruteForceOnlyOne(xs):
    rs = [Or(xs)]
    for (i,a) in enumerate(xs):
        for (j,b) in enumerate(xs):
            if (i != j):
                rs.append( Implies(a, -b) )
    return And(rs)

def sumOfVariables(S, T, variables, maxk):
    # S(k, i) = there are at leat k truthy values
    #           among the first i variables
    # T(k)   = there are k truthy values
    
    n = len(variables)
    
    if(maxk is None): maxk = n

    rules = []

    ## S

    for i in range(0, n+1):
        rules.append( S(0, i) )

    for k in range(1, maxk+1):
        rules.append( -S(k, 0) )
        for i in range(1, n+1):
            rules.append(Equivalent(
                S(k, i),
                Or([
                    S(k, i-1),
                    And([ variables[i-1], S(k-1, i-1) ])
                ])
            ))

    ## T
    for k in range(0, maxk):
        rules.append(Equivalent(
            T(k), And([S(k, n), -S(k+1, n)])
        ))

    rules.append(Equivalent(
        T(maxk), S(maxk,n)
    ))

    return And(rules)

def vectorLessThenOrEqual(E, xs, ys):
    # n = len(xs) = len(ys)
    # E_i, i in [0,n] :=  xs[0:i] == ys[0:i]

    n = len(xs)

    if(len(ys) != n):
        raise Exception('Imcompatible vector lengths')

    rules = []

    ## Eq
    rules.append( +E(0) )
    for (i, (x, y)) in enumerate1(zip(xs, ys)):
        rules.append(Equivalent(
            E(i),
            And([ E(i-1), Equivalent(x, y) ])
        ))

    ## x < y
    for i in range(0,n):
        rules.append(
            Implies( E(i), Implies(xs[i], ys[i]) )
        )

    return And(rules)

###########
# Neighbours
###########

# Radial Hexagonal coordinates
# (r, c, d)
# r = distance from center
# c = vértice associado
# d = indice no lado (0 é o vértice, r-1 é o último)
#
#  |0__ |1
#  /      \
# 5        --
# --        2
#  \       /
#   4| __3|
   
#Directional Hexagonal coordinates
# (a,b)
# / a
# - b
#
#    (-1,-1) (-1, 0)
# ( 0,-1) ( 0, 0) ( 0, 1)
#    ( 1, 0) (1, 1)

# Positional coordinates, as they come
# from the input:
#
#  0 1
# 2 3 4
#  5 6

#Simmetry functions

def reflect_0_5((r, c, d)):
    if r == 0 and c == 0 and d == 0:
        return (0,0,0)
    else:
        if d == 0:
            return (r, 5-c, 0)
        else:
            return (r, (10-c)%6, r-d)        

def clockwise_rotate(n, (r,c,d)):
    if r == 0 and c == 0 and d == 0:
        return (0,0,0)
    else:
        return (r, (c+n)%6, d)

class HexTopology(object):
    def __init__(self, side):
        self.rcd_to_m = {}
        self.ab_to_m = {}

        self.side = side
        
        self.rcds = []
        self.abs = []
        self.ps = []

        m_ = [0]
        a_ = [None]
        b_ = [None]
        def match(rcd):
            m = m_[0]
            ab = (a_[0], b_[0])

            self.abs.append(ab)
            self.rcds.append(rcd)
            self.ps.append(m)

            self.rcd_to_m[rcd] = m
            self.ab_to_m[ab] = m

            m_[0] += 1
            b_[0] += 1
            #print ''.join(map(str,rcd)),

        radius = side-1 

        #top half
        for r in range(radius, 0, -1):
            a_[0] = -r
            b_[0] = -radius
            #print '  '*r,
            for i in range(0, radius-r):
                match((radius-i, 5, r))
            for i in range(r):
                match((r, 0, i))
            for i in range(0, radius+1-r):
                match((r+i, 1, i))
            #print 

        #middle divider
        a_[0] = 0
        b_[0] = -radius
        #print '',
        for r in range(radius, 0, -1):
            match((r, 5, 0))
        if(radius >= 0):
            match((0,0,0))
        for r in range(1, radius+1):
            match((r, 2, 0))
        #iprint 

        #lower half
        for r in range(1, radius+1):
            a_[0] = r
            b_[0] = -(radius-r)
            #print '  '*r,
            for i in range(0, radius+1-r):
                match((radius-i, 4, radius-r-i))
            for i in range(r):
                match((r, 3, r-i-1))
            for i in range(0, radius-r):
                match((r+1+i, 2, r))
            #print
        
    def print_in_hex(self, xs):
        xs = list(reversed(xs))
        side = self.side
        lines = []

        def show(n):
            return ('.' if n is None else str(n))

        #upper half (with middle line)
        for (i,a) in enumerate(range(1-side, 0+1)):
            line = []
            for b in range(1-side, i+1):
                line.append(show(xs.pop())) 
            lines.append( ' '*(side-i-1) + ' '.join(line) )
        #lower half (without middle line)
        for (i,a) in enumerate1(range(1, side)):
            line = []
            for b in range(1-side+i, side):
                line.append(show(xs.pop()))
            lines.append( ' '*i + ' '.join(line) )
        return '\n'.join(lines)

    def hex_adjacency_graph(self):
        adj_list = {}
       
        def add(m, n):
            adj_list[m].append(n)

        def is_adj(m,n):
            add(m, n)
            if not adj_list.has_key(n):
                adj_list[n] = []
            add(n, m)

        for h in self.abs:
            (a,b) = h

            if not adj_list.has_key(h):
                adj_list[h] = []

            for h_ in [
                    (a-1, b-1),  (a-1, b),
                                    (a,  b+1)
                    ]:
                if self.ab_to_m.get(h_) is not None:
                    is_adj(h, h_)

        for lst in adj_list.values():
            lst.sort()

        return adj_list

    def pos_adjacency_graph(self):
        adj_list = {}

        for (k, vs) in self.hex_adjacency_graph().iteritems():
            adj_list[ self.ab_to_m[k] ] =\
                [ self.ab_to_m[v] for v in vs]
        
        return adj_list

    def simmetries(self):
        simmetries = []

        def add_sim(rcds):
            simmetries.append([
                self.rcd_to_m[rcd]
                for rcd in rcds
            ])

        for n in range(6):
            add_sim([
                clockwise_rotate(n, rcd)
                for rcd in self.rcds
            ])
       
        for n in range(6):
            add_sim([
                reflect_0_5(clockwise_rotate(n, rcd))
                for rcd in self.rcds
            ])

        return simmetries

###########
# Input
###########

import re

class ProblemInput(object):
    def __init__(self, side, counts, blocked_positions, fixed_positions, lines):
        self.side = side
        self.counts = counts
        self.blocked_positions = blocked_positions
        self.fixed_positions = fixed_positions
        self.lines = lines

    def print_to_stdout(self):
        print self.side
        for line in self.lines:
            print line,

def read_input(fil):
    side = int(fil.readline())
    counts = [0]*7
    blocked_positions = []
    fixed_positions = []

    lines = []

    m=0
    for line in fil:
        lines.append(line)
        for match in re.finditer(r'(\+?)(\.|\d)', line):
            locked = (match.group(1) == '+')
            n = (None if match.group(2) == '.' else int(match.group(2)))
            
            if n is not None:
                counts[n] += 1

            if locked:
                if n is None:
                    blocked_positions.append(m)
                else:
                    fixed_positions.append( (m, n) )

            m += 1

    return ProblemInput(
        side,
        assoc_list(counts),
        blocked_positions,
        fixed_positions,
        lines
    )

####
# Create clauses
####

class SATFormulation(object):
    def __init__(self, board_input, ruleset, topology, clauses):
        self.board_input = board_input
        self.ruleset  = ruleset
        self.topology = topology
        self.clauses  = clauses

def SAT_formulation_from_board_input(board_input):
    
    ruleset = RuleSet()
    topology = HexTopology(board_input.side)

    # Schema
    ########

    slot_range  = (0, topology.ps[-1])
    slot_count = len(topology.ps)
    
    value_range = (0, 6)
    
    # (m,n) = There is an n-valued tile at slot m
    #         The m-th slot has n occupied neighbors
    Placement = ruleset.VarSet('P', [slot_range, value_range])

    # (m) = is the m-th tile occupied?
    Occupied = ruleset.VarSet('O', [slot_range])

    # (m)(k,i) = m-th slot has k occupied slots among its first i neighbours
    NeighbourPartialSum = []

    # (m)(k) = m-th slot has k occupied slots among its neighbours
    NeighbourSum = []


    # (n)(k,i) = there are k n-valued tiles on the first i slots?
    TilePartialSum = []

    # Rules
    #######

    print '== Creating CNF description of level'

    rules = []

    print 'Creating level-state rules...'
 
    for (m, n) in board_input.fixed_positions:
        rules.append( +Placement(m, n) )
   
    for m in board_input.blocked_positions:
        rules.append( -Occupied(m) )

    print 'Creating tile placement rules...'

    for m in topology.ps:
        rules.append( BruteForceOnlyOne(
            [-Occupied(m)] + [+Placement(m,n) for n in range(7)]
        ))

    adj_graph = topology.pos_adjacency_graph()

    print 'Constraining number of neighbour of occupied tiles...'

    for m in topology.ps:
        vs  = adj_graph[m]
        nvs = len(vs)
        
        NPSum = ruleset.VarSet('Nps_'+str(m),[
            (0, nvs), (0, nvs) ])
        NeighbourPartialSum.append(NPSum)

        NSum = ruleset.VarSet('Ns_'+str(m), [
            (0, nvs) ])
        NeighbourSum.append(NSum)

        rules.append(sumOfVariables(
            NPSum, NSum,
            [+Occupied(v) for v in adj_graph[m]],
            None
        ))

        for n in range(0, nvs+1):
            rules.append(Implies(
                Occupied(m),
                Equivalent( +Placement(m,n), +NSum(n) )
            ))
        
        for n in range(nvs+1, 7):
            rules.append( -Placement(m,n) )

    print 'Creating constraints for the amount of tiles used...'

    empty_count = len(topology.ps) - sum([c for (_,c) in board_input.counts])

    # (k,i) = k empty slots among the first i slots
    EmptyPartialSum = ruleset.VarSet('Eps', [
        (0, empty_count+1), (0, slot_count) ])

    EmptySum = ruleset.VarSet('Es', [
        (0, empty_count+1)
    ])

    rules.append(sumOfVariables(
        EmptyPartialSum, EmptySum,
        [ -Occupied(m) for m in topology.ps ],
        empty_count + 1
    ))

    for n in range(0, 7):
        tile_count = assoc_get(n, board_input.counts)

        TPSum = ruleset.VarSet('Tps_'+str(n),[
            (0, tile_count+1), (0, slot_count) ])
        TilePartialSum.append(TPSum)

        TSum = ruleset.VarSet('Ts_'+str(n), [
            (0, tile_count+1) ])

        rules.append(sumOfVariables(
            TPSum, TSum,
            [ Placement(m,n) for m in topology.ps ],
            tile_count + 1
        ))
        rules.append( +TSum(tile_count) )

    print 'Adding simmetry-breaking rules...'

    def simmetry_is_preserved(xs, ys):
        xys = zip(xs, ys)

        for m in board_input.blocked_positions:
            m_ = assoc_get(m, xys)
            if m_ not in board_input.blocked_positions:
                #print m, 'to', m_, 'simmetry not found'
                return False

        for (m,v) in board_input.fixed_positions:
            m_ = assoc_get(m, xys)
            if (m_,v) not in board_input.fixed_positions:
                #print (m,v), 'to', (m_, v), 'simmetry not found'
                return False
        return True
    
    def vars_from_sim(ms):
        vs = []
        for m in ms:
            vs.append( +Occupied(m) )
            vs.extend([ +Placement(m,n) for n in range(0, 7) ])
        return vs

    simms = topology.simmetries()
    sim0 = simms[0]
    vsim0 = vars_from_sim(sim0)
    for (i,sim1) in enumerate1( simms[1:] ):
        if simmetry_is_preserved(sim0, sim1):
            print '  (Simmetry #%s found!)'%(i) 
            vsim1 = vars_from_sim(sim0)
            rules.append(vectorLessThenOrEqual(
                ruleset.VarSet('SimEq_'+str(i), [(0, len(vsim0))]),
                vsim0,
                vsim1
            ))

    print 'Converting rules to CNF form...'
    
    return SATFormulation(
        board_input,
        ruleset,
        topology,
        And(rules).to_cnf()
    )


def get_SAT_assignments(fil):
    assignments = []
    for line in fil:
        if 'UNSAT' in line.upper():
            return None
        for word in line.split():
            if re.match(r'-?\d+$', word):
                n = int(word)
                if n == 0:
                    return assignments
                else:
                    assignments.append(n)
    return assignments


def print_board_from_assignments(formulation, assignments):
    
    ruleset = formulation.ruleset
    topology = formulation.topology

    P = ruleset.get_varset_by_name('P')

    layout = [None for p in topology.ps]
    with open(NAMED_CNF_RESULT_FILE, 'w') as result:
        for lit in assignments:
            sgn = sign(lit)
            var = abs(lit)

            print >> result,  ruleset.cnfVarToLit(lit)
                
            if sgn > 0 and P.contains(var):
                (m,n) = P.valuesOfIndex(var)
                layout[m] = n

    print '=== Initial input: ==='
    formulation.board_input.print_to_stdout()

    print '=== Solution ==='
    print 
    print topology.print_in_hex(layout)

########
# Main
########

import sys

def main():

    if len(sys.argv) <= 1:
        print "usage: ./hexiom_solve.py [0-40]"
        exit(1)

    level_no = int(sys.argv[1])

    input_filename   = LEVEL_INPUT_FILENAME_PATTERN(level_no)
    cnf_in_filename  = SAT_INPUT_FILENAME_PATTERN(level_no)
    cnf_out_filename = SAT_OUTPUT_FILENAME_PATTERN(level_no)

    with open(input_filename, 'r') as fil:
        board_input = read_input(fil)

    print '== Level to solve== '
    board_input.print_to_stdout()

    formulation = SAT_formulation_from_board_input(board_input)

    print '=== Writing CNF to file ==='
    with open(cnf_in_filename, 'w') as fil:
        formulation.ruleset.print_cnf_file(formulation, fil )
    print '=== Done! Calling SAT solver now ==='

    SAT_SOLVE(cnf_in_filename, cnf_out_filename)

    with open(cnf_out_filename, 'r') as fil:
        assignments = get_SAT_assignments(fil)

    if assignments is None:
        print '*** Got UNSAT result! ***'
    else:
        print '** Solution found! ***'
        print_board_from_assignments(formulation, assignments)

if __name__ == '__main__':
    main()
