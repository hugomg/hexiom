
LEVEL_INPUT_FILENAME_PATTERN = (lambda n: 'levels/level%02d.txt'  % n )
SAT_INPUT_FILENAME_PATTERN   = (lambda n: 'sat_in/level%02d.cnf'  % n )
SAT_OUTPUT_FILENAME_PATTERN  = (lambda n: 'sat_out/level%02d.txt' % n )

NAMED_CNF_INPUT_FILE  = 'scratch/formula.txt'
NAMED_CNF_RESULT_FILE = 'scratch/result.txt'

import subprocess

def in_file_out_file(exe_name):
    ''' Run a minisat style solver'''
    def solve(infilename, outfilename):
        return subprocess.call(
            [exe_name, infilename, outfilename]
        )
    return solve

def in_file_out_pipe(exe_name):
    ''' Run a precosat style solver'''
    def solve(infilename, outfilename):
        with open(outfilename, 'w') as fil:
            return subprocess.call(
                [exe_name, infilename],
                stdout=fil
            )
    return solve

#SAT_SOLVE = in_file_out_pipe('./lingeling')
#SAT_SOLVE = in_file_out_file('./cryptominisat')
SAT_SOLVE = in_file_out_file('./glucose_static')
