import os
import re
import subprocess
import time

import numpy as np

from smt.src.solve_smtlib import SMTLIBsolver
from utils import write_solution


class SMTLIBsolverRot(SMTLIBsolver):

    def __init__(self, data, output_dir, timeout, solver):
        super().__init__(data, output_dir, timeout, solver)
        if output_dir == "":
            output_dir = "./smt/out/rot"
        self.output_dir = output_dir

    def solve_instance(self, instance, ins_num):
        _, self.max_width, self.circuits = instance
        self.circuits_num = len(self.circuits)

        widths, heights = ([i for i, _ in self.circuits], [j for _, j in self.circuits])

        cwd = os.getcwd()
        if self.solver == 'z3':
            self.file = self.instances_dir + "ins-" + str(ins_num) + ".smt2"
        elif self.solver == 'cvc5':
            os.chdir(cwd + "/smt")
            self.file = "instances_smtlib/" + "ins-" + str(ins_num) + ".smt2"

        solution, spent_time = self.set_constraints(widths, heights)

        if self.solver == 'cvc5':
            os.chdir(cwd)

        if solution is not None:
            self.parse_solution(solution)
            circuits_pos = self.evaluate()
            write_solution(self.output_dir, ins_num, ((self.max_width, self.plate_height), circuits_pos),
                           spent_time)
            return ins_num, ((self.max_width, self.plate_height), circuits_pos), spent_time
        else:
            write_solution(self.output_dir, ins_num, None, 0)
            return ins_num, None, 0

    def set_constraints(self, widths, heights):

        lower_bound = sum([heights[i] * widths[i] for i in range(self.circuits_num)]) // self.max_width
        upper_bound = sum(heights) - min(heights)

        areas_index = np.argsort([heights[i] * widths[i] for i in range(self.circuits_num)])
        areas_index = areas_index[::-1]
        biggests = areas_index[0], areas_index[1]

        for self.plate_height in range(lower_bound, upper_bound):
            lines = []

            if self.solver == 'z3':
                lines.append(f"(set-option :timeout {self.timeout * 1000})")
                lines.append("(set-option :smt.threads 4)")
            elif self.solver == 'cvc5':
                lines.append(f"(set-option :produce-models true)")

            lines.append("(set-logic AUFLIA)")

            # Decision Variables
            for i in range(self.circuits_num):
                lines.append(f"(declare-const x_{i} Int)")
                lines.append(f"(declare-const y_{i} Int)")
                lines.append(f"(declare-const width_{i} Int)")
                lines.append(f"(declare-const height_{i} Int)")
                lines.append(f"(declare-const rotation_{i} Bool)")

            # Domain
            lines += [f"(assert (and (>= x_{i} 0) (<= x_{i} (- {self.max_width} width_{i}))))"
                      for i in range(self.circuits_num)]
            lines += [f"(assert (and (>= y_{i} 0) (<= y_{i} (- {self.plate_height} height_{i}))))"
                      for i in range(self.circuits_num)]

            # Handling rotation
            for i in range(self.circuits_num):
                lines.append(f"(assert (ite rotation_{i} "
                             f"(and (= {widths[i]} height_{i}) (= {heights[i]} width_{i})) "
                             f"(and (= {widths[i]} width_{i}) (= {heights[i]} height_{i})))) ")
                lines.append(f"(assert (ite (= {widths[i]} {heights[i]}) "
                             f"(not rotation_{i}) "
                             f"(or rotation_{i} (not rotation_{i}))))")

            # Constraints

            # No Overlapping
            for i in range(self.circuits_num):
                for j in range(0, i):
                    lines.append(f"(assert (or "
                                 f"(<= (+ x_{i} width_{i}) x_{j}) "
                                 f"(<= (+ x_{j} width_{j}) x_{i}) "
                                 f"(<= (+ y_{i} height_{i}) y_{j}) "
                                 f"(<= (+ y_{j} height_{j}) y_{i})))")

            # Symmetry breaking : fix relative position of the two biggest rectangles
            lines.append(f'(assert (or '
                         f'(> x_{biggests[1]} x_{biggests[0]}) '
                         f'(and (= x_{biggests[1]} x_{biggests[0]}) (>= y_{biggests[1]} y_{biggests[0]}))))')

            # Cumulative over rows
            for u in range(self.plate_height):
                lines.append(
                    f"(assert (>= {self.max_width} (+ {' '.join([f'(ite (and (<= y_{i} {u}) (< {u} (+ y_{i} height_{i}))) width_{i} 0)' for i in range(self.circuits_num)])})))")

            # Result
            lines.append("(check-sat)")
            lines.append(
                f"(get-value ({' '.join([f'width_{i} height_{i} x_{i} y_{i}' for i in range(self.circuits_num)])}))")
            lines.append("(exit)")

            with open(self.file, "w") as f:
                for line in lines:
                    f.write(line + "\n")

            if self.solver == 'z3':
                bashCommand = f"z3 -smt2 {self.file}"
            elif self.solver == 'cvc5':
                bashCommand = f"cvc5 {self.file} --tlimit-per {self.timeout * 1000}"
            else:
                return None, 0
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            start_time = time.time()
            output, _ = process.communicate()
            time_spent = time.time() - start_time

            solution = output.decode('ascii')

            if solution.split("\r")[0] == 'sat':
                return solution, time_spent

            if solution.split("\r")[0] != 'sat' or time_spent >= 300:
                return None, time_spent

    def parse_solution(self, solution):
        if self.solver == 'z3':
            text = solution.split("\r")
            sat = re.compile(r'sat')
            text = [i for i in text if not sat.match(i)]
            text = [re.sub("\n", '', text[i]) for i in range(len(text))]
            text = [i for i in text if i != '']
            for i in range(len(text)):
                text[i] = re.sub("\(|\)", '', text[i])
                text[i] = text[i].split(" ")
                text[i] = [j for j in text[i] if j != '']

            self.w = [int(text[i][1]) for i in range(self.circuits_num * 4) if i % 4 == 0]
            self.h = [int(text[i][1]) for i in range(self.circuits_num * 4) if i % 4 == 1]
            self.x_positions = [int(text[i][1]) for i in range(self.circuits_num * 4) if i % 4 == 2]
            self.y_positions = [int(text[i][1]) for i in range(self.circuits_num * 4) if i % 4 == 3]
        elif self.solver == 'cvc5':
            text = solution.split("\n")
            text = [i for i in text if i != '']
            text = text[1:]
            for i in range(len(text)):
                text[i] = re.sub("\r", "", text[i])
                text[i] = re.sub("\(|\)", '', text[i])
                text[i] = text[i].split(" ")
            text = text[0]

            self.w = [int(text[i]) for i in range(self.circuits_num * 8) if i % 8 == 1]
            self.h = [int(text[i]) for i in range(self.circuits_num * 8) if i % 8 == 3]
            self.x_positions = [int(text[i]) for i in range(self.circuits_num * 8) if i % 8 == 5]
            self.y_positions = [int(text[i]) for i in range(self.circuits_num * 8) if i % 8 == 7]
